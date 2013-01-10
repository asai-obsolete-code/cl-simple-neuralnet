#|
  This file is a part of cl-simple-neuralnet project.
  Copyright (c) 2012 Masataro Asai (guicho2.71828@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-simple-neuralnet-test-asd
  (:use :cl :asdf))
(in-package :cl-simple-neuralnet-test-asd)

(defsystem cl-simple-neuralnet-test
  :author "Masataro Asai"
  :license ""
  :depends-on (:cl-simple-neuralnet
               :cl-test-more
			   :iterate
			   :vecto)
  :components ((:module "t"
                :components
                ((:file "test"))))
  :perform (load-op :after (op c) (asdf:clear-system c)))
