#|
This file is a part of cl-simple-neuralnet project.
Copyright (c) 2012 Masataro Asai (guicho2.71828@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-simple-neuralnet-test
  (:use :cl
        :cl-simple-neuralnet.core
		:cl-simple-neuralnet.utilities
		:annot
		:iterate
		:vecto
        :cl-test-more)
  (:shadow :terminate))
(in-package :cl-simple-neuralnet-test)

(declaim (optimize (debug 0) (space 0) (safety 0) (speed 3)))

(annot:enable-annot-syntax)

(defun fn (x y)
  (let ((x (d* pi x))
		(y (d* pi y)))
	(d+ 0.5d0
		(d* (dsin (d* 2.0d0 x))
			(dsin (d* 1.0d0 x))
			(dsin (d* 2.0d0 y))
			(dsin (d* 1.0d0 y))))))

(defun fn2 (x y)
  @type *desired-type* x
  @type *desired-type* y
  (d* (dsin (d* pi x)) (dsin (d* pi y))))


(defun around-p (x center width)
  (< (abs (- x center)) width))

(defun fill-method (z)
  (if (or (around-p (abs z) 0.2 0.01)
		  (around-p (abs z) 0.4 0.01)
		  (around-p (abs z) 0.6 0.01)
		  (around-p (abs z) 0.8 0.01))
	  (set-rgb-fill 0 0 0)
	  (set-rgba-fill 1 0 0 (abs z))))

(defun write-func-to-png (fn path)
  (with-canvas (:width 400 :height 400)
	(scale 400 400)
	(set-rgb-fill 1 1 1)
	(rectangle 0 0 1 1)
	(iter (for x0 from 0.0d0 to 1.0d0 by 1.0d-2)
		  (iter (for y0 from 0.0d0 to 1.0d0 by 1.0d-2)
				(rectangle x0 y0 1.0d-2 1.0d-2)
				(fill-method (funcall fn x0 y0))
				(fill-path)))
	(save-png path)))

(defun read-new-value ()
  (format t "Enter a new value: ")
  (multiple-value-list (eval (read))))

(defun bp-test (fn &optional (name "bp"))
  (let* ((nn (make-instance 'neural-network :nodes #(2 10 1))))
	(diag "~%w before leaning:~%~a"(w-of nn))
	(iter (with print-par = 130000)
		  (for total from 0)
		  (for i from 3 downto 0)
		  (iter (repeat print-par)
				(for x = (make-input (drandom 1.0d0)
									 (drandom 1.0d0)))
				(back-propagate
				 x (make-output-from-input fn x) nn))
		  
		  (for j = 
			   (/ (iter (repeat 100)
				   (for x = (make-input (drandom 1.0d0)
										(drandom 1.0d0)))
				   (summing
					(j-at x (make-output-from-input fn x) nn)))
					 100))
		  (for jj previous j)
		  (format t
				  "~%~ath bp: average J=~6f dJ=~6f"
				  (* total print-par) j (when jj (- j jj)))
		  (when (= (mod total 10) 0)
			(format t "~%drawing the learned function...")
			(write-func-to-png
			 (lambda (x0 y0)
			   (aref (propagate (make-input x0 y0) nn) 0))
			 (concatenate 'string name ".png"))
			(format t "~%drawing the original function...~%")
			(write-func-to-png fn (concatenate 'string name "-ans.png"))
			(diag "~%w after leaning:~%~a" (w-of nn)))
		  (restart-case
			  (when (= i 1)
				(error "iteration finished. what do you do?"))
			(add-iteration (new-i)
			  :interactive read-new-value
			  (setf i new-i))))))

(plan nil)
(deftest sigmoid1
  (ok (= 0.5d0 (sigmoid1 0.0d0)))
  (iter (repeat 10)
		(for x = (drandom most-positive-double-float))
		(ok (cond 
			  ((plusp x) (> (sigmoid1 x) 0.5d0))
			  ((zerop x) (= (sigmoid1 x) 0.5d0))
			  ((plusp x) (< (sigmoid1 x) 0.5d0))))))

(deftest j
  (let* ((nn (make-instance 'neural-network))
		 (x 1.0d0) (y 1.0d0)
		 (z0 '(1.0d0 1.0d0)))
	(diag (j-at (make-input x y) z0 nn))))

(deftest mostout-δ
  (ok (equalp (mostout-δ #(1.0d0) '(1.1d0))
			  (let* ((tk 1.1d0)
					(zk 1.0d0)
					(d (d* (d- tk zk)
						   zk (d- 1.0d0 zk))))
				(vector d)))))

(deftest hidden-δ
  (let ((out #(0.5d0 0.5d0))
		(w #2A((2.0d0) (2.0d0)))
		(d1 #(1.0d0)))
	(diag (hidden-δ out w d1))))

(defun fn-test (x y)
  (+ (* x x) (* y y)))

(deftest bp
  (let* ((x (make-input 1.0d0 1.0d0))
		 (z0 (make-output #'fn-test 1.0d0 1.0d0))
		 (nn (make-instance 'neural-network :nodes #(2 3 1)))
		 (w (w-of nn))
		 (nodes (nodes-of nn)))
	(ok (equalp x #(1.0d0 1.0d0)) "make-input")
	(ok (equalp z0 '(2.0d0)) "make-output")
	(is (length w) 2)
	(is (length nodes) 3)
	(iter
	  (with o = (layer-status-after-propagation x nn))
	  (is (length o) 3)
	  (with δ = (generate-all-δ w o z0))
	  (is (length δ) 2)
 
	  (for n below (length nodes))		;0,1,2
	  (for δn in δ)
	  (for on-1 in-vector o)
	  (for wn in-vector w)
	  (for (im jm) = (array-dimensions wn))

	  (diag δ)
	  (diag "n is ~a" n)
	  (is (length δn)
		  (aref nodes (1+ n)))
	  (is (length on-1)
		  (aref nodes n))
	  (is im (1+ (aref nodes n)))
	  (is jm (aref nodes (1+ n)))
	  (diag (array-dimensions wn))

	  (iter
		(for i below (1- im))
		(iter
		  (for j below jm)
		  (diag "~a ~a" i j)
		  (incf (aref wn i j)
				(d* +η+ (aref δn j) (aref on-1 i)))))
	  (iter
		(for j below jm)
		(incf (aref wn (1- im) j)
			  (d* +η+ (aref δn j) 1.0d0))))))

(finalize)
