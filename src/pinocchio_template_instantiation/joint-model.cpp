#include "simplex/fwd.hpp"
#include "pinocchio/multibody/joint/joint-generic.hpp"

template struct ::pinocchio::JointModelTpl<::simplex::context::Scalar, ::simplex::context::Options, ::pinocchio::JointCollectionDefaultTpl>;

template struct ::pinocchio::JointDataTpl<::simplex::context::Scalar, ::simplex::context::Options, ::pinocchio::JointCollectionDefaultTpl>;
