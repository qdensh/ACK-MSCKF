/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <ackermann_msckf/image_processor_ackermann_nodelet.h>

namespace ackermann_msckf {
void ImageProcessorAckermannNodelet::onInit() {
  img_processor_ackermann_ptr.reset(new ImageProcessorAckermann(getPrivateNodeHandle()));
  if (!img_processor_ackermann_ptr->initialize()) {
    ROS_ERROR("Cannot initialize Image Processor...");
    return;
  }
  return;
}

PLUGINLIB_EXPORT_CLASS(ackermann_msckf::ImageProcessorAckermannNodelet, nodelet::Nodelet);

} // end namespace ackermann_msckf

