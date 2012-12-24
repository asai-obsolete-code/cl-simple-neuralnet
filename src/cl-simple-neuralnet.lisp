#|
  This file is a part of cl-simple-neuralnet project.
  Copyright (c) 2012 Masataro Asai (guicho2.71828@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-simple-neuralnet
  (:use :cl
		:iterate
		:annot
		:annot.doc
		:annot.class
		:cl-simple-neuralnet.utilities)
  (:import-from :alexandria :copy-array))
(in-package :cl-simple-neuralnet)

(annot:enable-annot-syntax)

(declaim (optimize (debug 3)))

@export
(defun sigmoid (gain)
  @type *desired-type* gain
  (lambda (x)
	@type *desired-type* x
	(d/ (d+ 1.0d0 (dexp (d- (d* gain x)))))))

(setf (symbol-function 'sigmoid1) (sigmoid 0.5d0))
(export '(sigmoid1))

(defun randomize (ary limit)
  (iter (for i from 0 below (array-total-size ary))
		(setf (row-major-aref ary i)
			  (d- (drandom (d* 2.0d0 limit)) limit)))
  ary)

@export
@export-accessors
(defclass neural-network ()
  ((nodes :type (array fixnum (*))
		  :initform #(2 10 1)
		  :accessor nodes-of
		  :initarg :nodes)
   (w :type (array (array double-float (* *)) (*))
	:accessor weight-of
	:accessor w-of)))

(defmethod print-object ((nn neural-network) stream)
  (print-unreadable-object (nn stream :type t)
    (format stream "~a" (nodes-of nn))))


(defun make-weight (n1 n2)
  (make-array (list (1+ n1) n2) :element-type '*desired-type*))

@export
(defparameter *initial-randomization-weight-range* 1.0d-1)

(defmethod initialize-instance :after ((nn neural-network) &rest args)
  @ignore args
  (with-slots (w nodes) nn
	(iter
	  (with len = (length nodes))
	  (initially (setf w (make-array (1- len))))
	  (for n from 1 below len)
	  (for n-1 previous n initially 0)
	  (setf (aref w n-1)
			(randomize
			 (make-weight (aref nodes n-1) (aref nodes n))
			 *initial-randomization-weight-range*)))))

@export
(defun layer-input (l w)
  @type (simple-array *desired-type* 1) l
  @type (simple-array *desired-type* 2) w
  (iter 
	(with n1 = (1- (array-dimension w 0)))
	(with n2 = (array-dimension w 1))
	(with l2 = (make-array n2 :element-type '*desired-type*))
	(for j below n2)
	(setf (aref l2 j)
		  (d+ (iter (for i below n1)
					(summing (d* (aref w i j)
								 (aref l i))))
			  (aref w n1 j))) 			; * 1.0d0
	(finally (return l2))))

@export
(defun layer-output (l)
  @type (simple-array *desired-type* (*)) l
  (map '(simple-array *desired-type* (*)) #'sigmoid1 l))

@export
(defun propagate (input nn)
  (with-slots ((w-all w)) nn
	(reduce (lambda (stimula w)
			  (layer-output
			   (layer-input stimula w)))
			w-all :initial-value input)))

(defun sqdiff (a b)
  (d^2 (d- a b)))

@export
(defun j-at (x z0 nn &aux (z? (propagate x nn)))
  (reduce #'+ (map 'vector #'sqdiff z? z0)))

@export
(defun layer-status-after-propagation (input nn)
  (with-slots (w nodes) nn
	(iter (with len = (length nodes))
		  (with ys = (make-array len :fill-pointer 0))
		  (for stimula previous output initially input)
		  (for w-layer in-vector w)
		  (for output = (layer-output (layer-input stimula w-layer)))
		  (vector-push output ys)
		  (finally (return ys)))))

@export
@doc "the parameter for steepest descent method."
(defparameter +η+ 8.0d-2)

@export
(defun mostout-δ (z z0)
  (map 'vector
	   (lambda (zk tk)
		 (d* (d- tk zk)
			 zk (d- 1.0d0 zk)))
	   z z0))

@export
(defun hidden-δ (out w δ-1)
  (iter 
	(with dim-1 = (1- (array-dimension w 0)))
	(with δ = (make-array dim-1))
	(for oj in-vector out with-index j)
		(setf (aref δ j)
			  (d* (iter 
					(for δ-1k in-vector δ-1 with-index k)
					(summing (d* (aref w j k) δ-1k)))
				  oj (d- 1.0d0 oj)))
		(finally (return δ))))

@export
(defun generate-all-δ (w o z0)
  (nreverse
   (iter
	 (with δ-1 = (mostout-δ
				   (aref o (1- (length o)))
				   z0))
	 (for on
		  in-vector o
		  with-index n		;nth layer
		  downfrom (- (length o) 2))
	 (for wn in-vector w downto 0)
	 (for δ = (hidden-δ on wn δ-1))
	 (setf δ-1 δ)
	 (collecting δ))))

@export
(defun back-propagate (x z0 nn)
  (with-slots (w nodes) nn
	(iter
	  (with o = (layer-status-after-propagation x nn))
	  (with δ = (generate-all-δ w o z0))
	  (for δn in δ)
	  (for on-1 in-vector o)
	  (for wn in-vector w with-index n)
	  (for (im jm) = (array-dimensions wn))
	  (iter
		(for i below (1- im))
		(iter
		  (for j below jm)
		  (incf (aref wn i j)
				(d* +η+ (aref δn j) (aref on-1 i)))))
	  (iter
		(for j below jm)
		(incf (aref wn (1- im) j)
			  (d* +η+ (aref δn j) 1.0d0))))))

@export
(defun make-input (&rest args)
  (coerce args '(array *desired-type* 1)))

@export
(defun make-output (fn &rest args)
  (multiple-value-list (apply fn args)))

@export
(defun make-output-from-input (fn input)
  (multiple-value-list (apply fn (coerce input 'list))))


@export
@doc "
bp-teach (fn, nodes, &key iteration, nn) -> nn

`fn' : the target function. input-arguments* -> output-arguments*
`iteration' : a `fixnum'
`nodes' : ({ number-of-nodes-in-layer }*)
`number-of-nodes-in-layer' : a `fixnum'
`nn' : an instance of `neural-network'

let I = [0.0d0,1.0d0] .
function `fn' should accept n `double-float' arguments within I
and is expected to return m `double-float' arguments values.
n should match the first `fixnum' in the `nodes' , and m should
match the last `fixnum' in the `nodes'.

if `nn' is unspecified, `nodes' argument is used to create 
a new instance of `neural-network'
otherwise, `nodes' will be ignored and it will
conduct further teaching on `nn'.

`iteration' determines iteration number of
 back-propagation algorhithm, defaulted to 10000.
"
(defun bp-teach (fn nodes
				 &key
				 (iteration 10000)
				 nn)
  (let ((nn (or nn
				(make-instance
				 'neural-network
				 :nodes (coerce nodes 'vector)))))
	(print nn)
	(iter (repeat iteration)
		  (with dim = (car nodes))
		  (with input = (make-array dim :element-type '*desired-type*))
		  (iter (for xi in-vector input with-index i)
				(setf (aref input i) (random 1.0d0)))
		  (for z0 = (make-output-from-input fn input))
		  (back-propagate input z0 nn))
	nn))
		  
		  