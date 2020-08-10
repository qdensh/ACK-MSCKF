/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_NODELET_H
#define MSCKF_VIO_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ackermann_msckf/ackermann_msckf.h>

namespace ackermann_msckf {
class AckermannMsckfNodelet : public nodelet::Nodelet {
public:
  AckermannMsckfNodelet() { return; }
  ~AckermannMsckfNodelet() { return; }

private:
  virtual void onInit();
  AckermannMsckfPtr ackermann_msckf_ptr;
};
} // end namespace ackermann_msckf

#endif

