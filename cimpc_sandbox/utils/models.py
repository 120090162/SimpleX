from exceptiongroup import catch
import numpy as np
import pinocchio
import crocoddyl

import hppfcl
from hppfcl import CollisionObject

# 地面对象, 用于碰撞检测
ground_width = 100.0
ground_height = 100.0
ground_depth = 50
ground_geo = hppfcl.Box(ground_width, ground_height, ground_depth)
ground_pos = np.array([0., 0., -ground_depth/2])
ground_transform = hppfcl.Transform3f(np.eye(3), ground_pos)
ground_obj = CollisionObject(ground_geo, ground_transform)

import pinocchio.casadi as cpin
from casadi import SX, Function, jacobian, vertcat, MX

from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, Bounds
from scipy.linalg import block_diag
# from scipy import sparse

from pycontact import RaisimSolver, RaisimCorrectedSolver, ContactProblem, ContactSolverSettings


from line_profiler import profile

from .transforms import *
from .costs import set_lf_cost

class ContactModel:
    def __init__(self, state, contact_ids, friction):
        self.contact_ids = contact_ids
        self.cmodel = cpin.Model(state.pinocchio)
        self.cdata = self.cmodel.createData()
        self.qsym = SX.sym('q', state.pinocchio.nq-1) # cpin只允许SX
        
        self.contact_ids = contact_ids
        self.friction = friction
                
        # 符号微分的函数只用算一次
        self.auto_diff()
        
    @profile    
    def auto_diff(self):
        # 利用casadi计算frame Jacobian, M inverse, foot height的符号微分
        cmodel, cdata = self.cmodel, self.cdata
        qsym = self.qsym
        self.dJ_dq_funs = []
        self.dh_dq_funs = []
        
        for i in self.contact_ids:
            def frameJacobian(q_sym):
                qquat = csrpy_to_quat(q_sym[3:6])
                inp_q = vertcat(q_sym[:3],qquat,q_sym[6:])
                cpin.forwardKinematics(cmodel,cdata,inp_q)
                cpin.computeJointJacobians(cmodel,cdata,inp_q)
                J = cpin.getFrameJacobian(
                    cmodel, cdata, i, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                return J[:3,:]
            frameJsym = frameJacobian(qsym)
            frameJ_dq = jacobian(frameJsym, qsym)
            frameJ_dq_fun = Function('jacobian', [qsym], [frameJ_dq])
            self.dJ_dq_funs.append(frameJ_dq_fun) # frame Jacobian的微分
            
            def frameheight(q_sym):
                qquat = csrpy_to_quat(q_sym[3:6])
                inp_q = vertcat(q_sym[:3],qquat,q_sym[6:])
                cpin.forwardKinematics(cmodel,cdata,inp_q)
                return cdata.oMf[i].translation[2]
            heightsym = frameheight(qsym)
            height_dq = jacobian(heightsym, qsym)
            dh_dq_fun = Function('height_dq', [qsym], [height_dq])
            self.dh_dq_funs.append(dh_dq_fun) # foot height的符号微分
            
        def Minvfun(q_sym):
            qquat = csrpy_to_quat(q_sym[3:6])
            inp_q = vertcat(q_sym[:3],qquat,q_sym[6:])
            cpin.computeMinverse(cmodel, cdata, inp_q)
            return cdata.Minv
        Minvfunsym = Minvfun(qsym)
        dMinv_dq = jacobian(Minvfunsym,qsym)        
        self.dMinv_dq_fun = Function('jacobian', [qsym], [dMinv_dq]) # M inverse的符号微分
    
# 存robot/cost/actuation模型和中间变量用的
class DAD_contact(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None
       
        
class DAM_contact(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuationModel, costModel, contactModel, dt, rho=0):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, actuationModel.nu, costModel.nr
        )
        self.actuation = actuationModel
        self.costs = costModel
        self.contact = contactModel
        
        # 脚部建立hppfcl碰撞对象，用于和地面检测碰撞
        self.contact_ids = contactModel.contact_ids
        self.contact_radius = 0.1  # 碰撞检测范围, 要比脚大一些, 预防穿模
        geo = hppfcl.Sphere(self.contact_radius)
        trans = hppfcl.Transform3f(np.eye(3), np.zeros(3))
        self.contact_objects = [CollisionObject(geo, trans) for i in range(len(self.contact_ids))]
        
        # 接触计算相关参数
        self.contact_radius1 = 0.01428652  # 脚部半径, 冲量计算, 要接近脚的真实半径, 决定真实冲量, 防止穿模
        self.dt = dt
        self.friction = contactModel.friction
        self.rho = rho
        
        # contact problem求解超参
        self.contact_maxiter = 500
        self.contact_eps = 1e-4
        
        # BoxFDDP需要这些属性
        self.u_lb = -np.array(state.pinocchio.effortLimit[-12:])
        self.u_ub = np.array(state.pinocchio.effortLimit[-12:])
        
        # 设置不了，但是py里不影响, cpp不知道需不需要设置, 也是为了BoxFDDP
        # self.has_control_limits = True
    
    # 求解contact problem, 输入frame信息, 输出impulse
    @profile
    def solve_contact(self,G,g,friction,collision_ids):
        # G: J@Minv@J.T, frame的local质量矩阵
        # g: free velocity of contact frame, 不考虑接触的足端速度
        
        # 初始化
        lam0 = np.zeros(3*len(collision_ids))
        prob = ContactProblem(G,g,friction)
        
        # 可选不同求解器
        # solver = RaisimCorrectedSolver()
        solver = RaisimSolver() # 快一丢
        # solver = CCPADMMSolver() # 一样快，但最后结果更差
        # solver = CCPPGSSolver() # 更快，但最后结果更差
        
        # 求解
        solver.setProblem(prob)
        settings = ContactSolverSettings()
        settings.max_iter_ = self.contact_maxiter
        settings.th_stop_ = self.contact_eps
        settings.rel_th_stop_ = self.contact_eps
        hasConverged = solver.solve(prob, lam0, settings)
        assert hasConverged
        impulse = solver.getSolution().copy()
        
        return impulse
    
    # 碰撞检测, 先fcl检测需不需要求contact problem, 如果需要则调用solve_contact求impulse
    @profile
    def collision_test(self,data,tau,v):
        
        # 更新fcl碰撞对象, 即更新足端球的位置
        for i, contact_id in enumerate(self.contact_ids):
            oMf = data.pinocchio.oMf[contact_id]
            pos = oMf.translation
            rot = oMf.rotation
            self.contact_objects[i].setTransform(hppfcl.Transform3f(rot, pos))
            
            # set_lf_cost(self.costs, self.state, self.actuation, contact_id ,pos[2])
        
        # 检测碰撞
        collision_ids = []
        normal_trans = []
        height = []
        friction = []
        for i in range(len(self.contact_objects)):
            result = hppfcl.CollisionResult() # 必须清空...
            req = hppfcl.CollisionRequest()   
            if hppfcl.collide(self.contact_objects[i], ground_obj, req, result):
                # contact = result.getContacts()[0]
                collision_ids.append(self.contact_ids[i])
                
                # current = np.array([0,0,1])
                # target = -contact.normal
                # normal_trans.append(pinocchio.Quaternion.FromTwoVectors(current,target).matrix())
                
                # height.append(contact.pos[2])
                height.append(data.pinocchio.oMf[self.contact_ids[i]].translation[2])
                friction.append(self.friction[i])
                
        height = np.array(height)
        data.collision_ids = collision_ids
        
        if collision_ids == []:
            data.real_collision = False
            return False
        
        # 有碰撞就开始计算
        # 计算G和g
        col_num = len(collision_ids)
        Jk = [
            # normal_trans[i]@
            pinocchio.getFrameJacobian(
                    self.state.pinocchio,
                    data.pinocchio,
                    collision_ids[i],
                    pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:3,:] for i in range(col_num)
        ]
        
        data.M = data.pinocchio.M
        data.Minv = data.pinocchio.Minv
    
        ddqf = np.dot(data.Minv, ( tau - data.pinocchio.nle ))
        J = np.vstack(Jk)
        data.contactpreb = (ddqf*self.dt+v)
                
        G=J@data.Minv@J.T
        g=J@data.contactpreb
        
        # 引入高度信息防止穿模
        g[2::3] += (height-self.contact_radius1)/self.dt
        impulse = self.solve_contact(G,g,friction,collision_ids)
        
        
        # 接触点分类
        slide_ids = []
        Es = []
        clamping_ids = []
        for i in range(col_num):
            if impulse[3*i+2] < self.contact_eps: # separate
                continue
            if (self.friction[i]*impulse[3*i+2])**2-impulse[3*i]**2-impulse[3*i+1]**2 < self.contact_eps: # slide
                slide_ids.append(i)
                Es.append(np.array([impulse[3*i]/impulse[3*i+2],impulse[3*i+1]/impulse[3*i+2],1])[:,np.newaxis])
            else:
                clamping_ids.append(i)
        
        # 储存微分的时候要用的数据
        J_ = []
        contact_impulse = []
        Jleft = []
        Jright = []
        h = []
        if clamping_ids != []:
            mask = sum([list(range(3*i, 3*i + 3)) for i in clamping_ids],[])
            Jc = J[mask,:]
            # Jc = np.vstack([J[(3*i):(3*i+3),:] for i in clamping_ids])
            impulsec = impulse[mask]
            # impulsec = np.hstack([impulse[(3*i):(3*i+3)] for i in clamping_ids])
            Jleft.append(Jc)
            Jright.append(Jc)
            J_.append(Jc)
            contact_impulse.append(impulsec)
            # h.append(np.hstack( [np.array([0,0,height[i]]) for i in clamping_ids] ))
            h += [np.array([0,0,height[i]]) for i in clamping_ids]
            
            data.Jc = Jc
            data.impulsec = impulsec
        if slide_ids != []:
            mask1 = sum([list(range(3*i+2, 3*i + 3)) for i in slide_ids],[])
            mask2 = sum([list(range(3*i, 3*i + 2)) for i in slide_ids],[])
            mask3 = sum([list(range(3*i, 3*i + 3)) for i in slide_ids],[])
            Es = block_diag(*Es)
            data.Es = Es
            Jsn = J[mask1,:]
            Jst = J[mask2,:]
            Js = J[mask3,:]
            impulsesn = impulse[mask1]
            impulsest = impulse[mask2]
            Jleft.append(Jsn)
            Jright.append(Es.T@Js)
            J_.append(Jsn)
            J_.append(Jst)
            contact_impulse += [impulsesn.flatten(),impulsest] # 要和J_对齐
            
            # h.append(np.hstack( [np.array([height[i]]) for i in slide_ids] ))
            h += [np.array([height[i]]) for i in slide_ids]
            
            data.Jsn = Jsn
            data.Jst = Jst
            data.impulsesn = impulsesn
            data.impulsest = impulsest
        
        if slide_ids+clamping_ids == []:
            data.real_collision = False
            return False
        
        data.h = np.concatenate(h)
        Jleft = np.vstack(Jleft)
        Jright = np.vstack(Jright)
        
        A = Jleft@data.Minv@Jright.T
        b = Jleft@(ddqf*self.dt+v)
        
        data.contactJleft = Jleft
        data.contactJright = Jright
        data.contactJ = np.vstack(J_)
        data.impulse = np.concatenate(contact_impulse)
        
        # 有rho则A要换
        D = data.impulse.copy()
        if slide_ids == []:
            D[::3] = 1
            D[1::3] = 1
            D = self.rho/D**2
            D[::3] = 0
            D[1::3] = 0
        else:
            snum = len(slide_ids)
            D = D[:(-2*snum)]
            D[:(-snum):3] = 1
            D[1:(-snum):3] = 1
            D = self.rho/D**2
            D[:(-snum):3] = 0
            D[1:(-snum):3] = 0
        
        try:
            Ainv = np.linalg.inv(A + np.diag(D))
        except Exception as e:
            Ainv = np.linalg.pinv(A + np.diag(D))
        
        data.slide_ids = slide_ids
        data.clamping_ids = clamping_ids
        data.contactAinv = Ainv
        data.contactb = b
        data.effect = data.contactJ.T@data.impulse/self.dt # 接触力=冲量/dt
        data.real_collision = True
        return True
    
    @profile
    def calc(self, data, x, u=None):
        if u is None: # 最后那一步N默认u=None
            q, v = x[: self.state.nq], x[-self.state.nv :]
            if v[2] < -q[2]/self.dt:
                v[2] = -q[2]/self.dt
                
            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)            
            self.costs.calc(data.costs, x)
            data.cost = data.costs.cost
        else:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            self.actuation.calc(data.actuation, x, u) # float base
            tau = data.actuation.tau
                        
            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            pinocchio.computeMinverse(self.state.pinocchio, data.pinocchio, q)
            pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
            
            collision = self.collision_test(data, tau, v)
            
            # Computing the dynamics using ABA, impulse的效果直接加到tau
            if collision:
                data.xout[:] = pinocchio.aba(
                self.state.pinocchio, data.pinocchio, q, v, tau+data.effect
                )
            else:
                data.xout[:] = pinocchio.aba(
                self.state.pinocchio, data.pinocchio, q, v, tau
                )
            
            # Computing the cost value and residuals
            pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
            pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
            
            self.costs.calc(data.costs, x, u)
            data.cost = data.costs.cost
    
    @profile
    def calcDiff(self, data, x, u=None):
        if u is None:
            self.costs.calcDiff(data.costs, x)
        else:
            nq, nv = self.state.nq, self.state.nv
            q, v = x[:nq], x[-nv:]
            # Computing the actuation derivatives
            self.actuation.calcDiff(data.actuation, x, u)
            tau = data.actuation.tau
            # Computing the dynamics derivatives
            if not data.real_collision:
                # Computing the cost derivatives
                pinocchio.computeABADerivatives(
                    self.state.pinocchio, data.pinocchio, q, v, tau
                )
                ddq_dq = data.pinocchio.ddq_dq
                ddq_dv = data.pinocchio.ddq_dv
                data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + data.pinocchio.Minv@data.actuation.dtau_dx
                data.Fu[:, :] = data.pinocchio.Minv@data.actuation.dtau_du
                self.costs.calcDiff(data.costs, x, u)
                return
            
            
            pinocchio.computeABADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, tau+data.effect
            )
            ddq_dq = data.pinocchio.ddq_dq
            ddq_dv = data.pinocchio.ddq_dv
            data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + data.Minv@data.actuation.dtau_dx
            data.Fu[:, :] = data.Minv@data.actuation.dtau_du
            
            # 复用aba的结果来计算db_dq db_dv, b就是free velocity
            pinocchio.computeABADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, tau
            )
            ddq_dq = data.pinocchio.ddq_dq + data.Minv@data.actuation.dtau_dx[:,:nv]
            ddq_dv = data.pinocchio.ddq_dv + data.Minv@data.actuation.dtau_dx[:,-nv:]
            
            # 从data拿必要的数据
            qrpy = quat_to_rpy(q[3:7])
            qrpy = np.concatenate([q[:3],qrpy,q[7:]],axis=0)
            contactJleft = data.contactJleft
            contactJright = data.contactJright
            Minv = data.Minv
            contactAinv = data.contactAinv
            contactJ = data.contactJ
            impulse = data.impulse
            collision_ids = data.collision_ids
            slide_ids = data.slide_ids
            clamping_ids = data.clamping_ids
            
            # 根据接触点分类算Jacobian和foot height的梯度
            dJc_dq = []
            dJsn_dq = []
            dJst_dq = []
            dJsright_dq = []
            dh_dq = []
            for fid in clamping_ids+slide_ids:
                i = self.contact_ids.index(collision_ids[fid])
                frameJ_dq_fun = self.contact.dJ_dq_funs[i]
                dJ_dqi = (frameJ_dq_fun.call([qrpy])[0]).full().reshape((nv,3,-1)).transpose((2,1,0))
                
                dh_dq_fun = self.contact.dh_dq_funs[i]
                dh_dq_i = (dh_dq_fun.call([qrpy])[0]).full().reshape((1,-1))
                
                if fid in clamping_ids:
                    dJc_dq.append(dJ_dqi)
                    dh_dq_zero = np.zeros_like(dh_dq_i)
                    dh_dq.append(np.concatenate([dh_dq_zero,dh_dq_zero,dh_dq_i],axis=0))
                else:
                    k = slide_ids.index(fid)
                    dJsn_dq.append(dJ_dqi[:,2:3,:])
                    dJst_dq.append(dJ_dqi[:,:2,:])
                    dJsright_dq.append(
                        data.Es[(3*k):(3*k+3),k:(k+1)].T[np.newaxis,:] @ dJ_dqi
                    )
                    dh_dq.append(dh_dq_i)
                
            dJ_dq_left = np.concatenate(dJc_dq+dJsn_dq,axis=1)
            dJ_dq_right = np.concatenate(dJc_dq+dJsright_dq,axis=1)
            dJ_dq = np.concatenate(dJc_dq+dJsn_dq+dJst_dq,axis=1)
            dh_dq = np.concatenate(dh_dq,axis=0)
            
            # Minv的梯度
            dMinv_dq_fun = self.contact.dMinv_dq_fun
            dMinv_dq = (dMinv_dq_fun.call([qrpy])[0]).full().reshape((nv,nv,-1)).transpose((2,1,0))
            
            dA_dq = dJ_dq_left@((Minv@contactJright.T)[np.newaxis,:]) + (contactJleft@Minv)[np.newaxis,:]@dJ_dq_right.transpose((0,2,1)) + (contactJleft[np.newaxis,:])@dMinv_dq@(contactJright.T[np.newaxis,:])
            
            db_dq = (dJ_dq_left@(data.contactpreb[np.newaxis,:,np.newaxis])).squeeze(2).T + contactJleft@ddq_dq*self.dt
            db_dv = contactJleft@(ddq_dv*self.dt+np.eye(nv))
            db_dtau = contactJleft@Minv*self.dt
            
            # impulse梯度
            if slide_ids == []:
                dlambda_dq = -contactAinv@( (dA_dq@impulse[np.newaxis,:,np.newaxis]).squeeze(2).T )
            else:
                dlambda_dq = -contactAinv@( (dA_dq@impulse[np.newaxis,:-(2*len(slide_ids)),np.newaxis]).squeeze(2).T )
            dlambda_dq += -contactAinv@(db_dq+dh_dq/self.dt)
                
            dlambda_dv = -contactAinv@db_dv
            dlambda_dtau = -contactAinv@db_dtau
            if slide_ids != []:
                Es = data.Es
                snum = len(slide_ids)
                Est = np.concatenate([Es[(3*i):(3*i+2),:] for i in range(snum)])
                dlambda_dq = np.vstack([dlambda_dq, Est@dlambda_dq[-snum:,:]])
                dlambda_dv = np.vstack([dlambda_dv, Est@dlambda_dv[-snum:,:]])
                dlambda_dtau = np.vstack([dlambda_dtau, Est@dlambda_dtau[-snum:,:]])
            
            # 最后加到原来的aba结果中
            Fq = Minv@ ( (dJ_dq.transpose(0,2,1)@(impulse[np.newaxis,:,np.newaxis])).squeeze(2).T + contactJ.T@dlambda_dq )/self.dt
            Fv = Minv@contactJ.T@dlambda_dv/self.dt
            Ftau = Minv@contactJ.T@dlambda_dtau/self.dt
            
            data.Fx[:, :] += np.hstack([Fq, Fv])
            data.Fu[:, :] += Ftau@data.actuation.dtau_du
            
            self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = DAD_contact(self)
        return data

# 构造one shoot problem
def IAM_shoot(N, state, actuation, costs, contact, DT, rho=0):
    assert N>1
    dmodelr = DAM_contact(
        state, actuation, costs[0], contact, DT, rho
    )
    dmodelt = DAM_contact(
        state, actuation, costs[1], contact, DT, rho
    )
    actionmodels = [crocoddyl.IntegratedActionModelEuler(dmodelr, DT)] * N + [crocoddyl.IntegratedActionModelEuler(dmodelt, 0.0)]
    return actionmodels



