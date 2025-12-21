// PINOCCHIO
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/fwd.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/algorithm/constraints/coulomb-friction-cone.hpp>
#include <pinocchio/algorithm/contact-cholesky.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/algorithm/contact-jacobian.hpp>
#include <pinocchio/algorithm/delassus-operator-base.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

#include <pinocchio/algorithm/admm-solver.hpp>
#include <pinocchio/algorithm/pgs-solver.hpp>
#include <pinocchio/algorithm/proximal.hpp>

#include <pinocchio/collision/broadphase-manager.hpp>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/multibody/geometry-object.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <pinocchio/parsers/mjcf.hpp>

#include <pinocchio/spatial/se3.hpp>

// EIGEN
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
