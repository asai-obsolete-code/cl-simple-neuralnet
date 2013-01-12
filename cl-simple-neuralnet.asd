#|
  This file is a part of cl-simple-neuralnet project.
  Copyright (c) 2012 Masataro Asai (guicho2.71828@gmail.com)
|#

#|
  a simple implementation of multiple-layered neural network.

  Author: Masataro Asai (guicho2.71828@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-simple-neuralnet-asd
  (:use :cl :asdf))
(in-package :cl-simple-neuralnet-asd)

(defsystem cl-simple-neuralnet
  :version "0.1"
  :author "Masataro Asai"
  :license ""
  :depends-on (:cl-annot :iterate :alexandria)
  :components ((:module "src"
				:serial t
                :components
                ((:file "typed-ops")
				 (:file "core")
				 (:file :package))))
  :description "a simple implementation of multiple-layered neural network."
  :long-description
  #.(with-open-file (stream (merge-pathnames
                             #p"README.markdown"
                             (or *load-pathname* *compile-file-pathname*))
                            :if-does-not-exist nil
                            :direction :input)
      (when stream
        (let ((seq (make-array (file-length stream)
                               :element-type 'character
                               :fill-pointer t)))
          (setf (fill-pointer seq) (read-sequence seq stream))
          seq)))
  :in-order-to ((test-op (load-op cl-simple-neuralnet-test))))
