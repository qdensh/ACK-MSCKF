/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <ackermann_msckf/ackermann_msckf_nodelet.h>

namespace ackermann_msckf {
void AckermannMsckfNodelet::onInit() {
  ackermann_msckf_ptr.reset(new AckermannMsckf(getPrivateNodeHandle()));
  if (!ackermann_msckf_ptr->initialize()) {
    ROS_ERROR("Cannot initialize MSCKF VIO...");
    return;
  }
  return;
}

PLUGINLIB_EXPORT_CLASS(ackermann_msckf::AckermannMsckfNodelet, nodelet::Nodelet);

} // end namespace ackermann_msckf

