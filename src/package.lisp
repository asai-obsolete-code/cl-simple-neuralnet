



(in-package :cl-user)
(defpackage cl-simple-neuralnet
  (:use :cl
		:cl-simple-neuralnet.core)
  (:export :*initial-randomization-weight-range*
		   :+η+
		   :sigmoid
		   :sigmoid1
		   :neural-network
		   :propagate
		   :back-propagate
		   :make-input
		   :make-output
		   :make-output-from-input
		   :bp-teach))
