/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef IMAGE_PROCESSOR_NODELET_H
#define IMAGE_PROCESSOR_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ackermann_msckf/image_processor_ackermann.h>

namespace ackermann_msckf {
class ImageProcessorAckermannNodelet : public nodelet::Nodelet {
public:
  ImageProcessorAckermannNodelet() { return; }
  ~ImageProcessorAckermannNodelet() { return; }

private:
  virtual void onInit();
  ImageProcessorAckermannPtr img_processor_ackermann_ptr;
};
} // end namespace ackermann_msckf

#endif

