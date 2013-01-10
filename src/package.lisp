
(in-package :cl-user)
(defpackage cl-simple-neuralnet
  (:use :cl
		:cl-simple-neuralnet.core)
  (:export
   :sigmoid :sigmoid1 :neural-network
   :nodes-of :w-of :weight-of
   :*initial-randomization-weight-range*
   :propagate
   :j-at
   :+η+
   :back-propagate
   :make-input
   :make-output
   :make-output-from-input
   :bp-teach))
