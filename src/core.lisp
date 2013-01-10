#|
  This file is a part of cl-simple-neuralnet project.
  Copyright (c) 2012 Masataro Asai (guicho2.71828@gmail.com)
|#


(in-package :cl-user)
(defpackage cl-simple-neuralnet.core
  (:use :cl
		:iterate
		:annot
		:annot.doc
		:annot.class
		:cl-simple-neuralnet.utilities)
  (:import-from :alexandria :copy-array))
(in-package :cl-simple-neuralnet.core)

(annot:enable-annot-syntax)

;; (declaim (optimize (debug 3)))

(declaim (optimize (debug 0) (space 0) (safety 0) (speed 3)))

@export
@doc "as the name says."
(defparameter *initial-randomization-weight-range* 0.5d0)

@export
@doc "the parameter for steepest descent method."
(defparameter +η+ 8.0d-2)


@export
(defun sigmoid (gain)
  @type *desired-type* gain
  (lambda (x)
	@type *desired-type* x
	(d/ (d+ 1.0d0 (dexp (d- (d* gain x)))))))

(setf (symbol-function 'sigmoid1) (sigmoid 0.5d0))
(export '(sigmoid1))

(defun sigmoid-inv1 (y)
  (dlog (d/ y (d- y 1.0d0))))

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
  ;; (assert (= (1- (array-dimension w 0))
  ;; 			 (length l)))
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
@doc "入力層以外のノードの出力をすべてあつめて返す。すなわちn=2から"
(defun layer-status-after-propagation (input nn)
  (with-slots (w nodes) nn ;; 数が重要！
	(iter (with len = (length nodes)) ;; 層の数3
		  (with ys = (make-array len)) ;; 出力の数も3
		  ;; (assert (= (1- (length nodes)) (length w)))
		  (for stimula previous output initially input)
		  (for w-layer in-vector w with-index n) ;; 重みの数は2段だけ
		  (for output = (layer-output (layer-input stimula w-layer)))
		  (setf (aref ys n) stimula) ;; 入力層はそのまま出力
		  (finally 
		   (setf (aref ys (1- len)) output) ;; 最後の出力
		   ;(break "~@{~a~}" ys n)
		   (return ys))))) ;; 出力の数は3


@export
(defun mostout-δ (z z0)
  ;; (assert (= (length z) (length z0)))
  (map 'vector
	   (lambda (zk tk)
		 (d* (d- tk zk)
			 zk (d- 1.0d0 zk)))
	   z z0))

@export
(defun hidden-δ (out w δ-1)
  ;; out :: 一つ前の層の出力    1.0の要素はない
  ;; δ-1 :: 一つ次の層の誤差項. 1.0の要素はない（誤差0だから)
  ;; δ   :: 一つ前の層の誤差項. 1.0の要素はない（誤差0だから)
  ;; (assert (= (array-dimension w 1) (length δ-1)))
  ;; (assert (= (1- (array-dimension w 0)) (length out)))
  ;; つねに1のノードがあるため
  (iter 
	(with δ = (make-array (1- (array-dimension w 0))))
	;; (assert (= (1- (array-dimension w 0)) (length δ)))
	(for oj in-vector out with-index j)
	(setf (aref δ j)
		  (d* (iter 
				(for δ-1k in-vector δ-1 with-index k)
				(summing (d* (aref w j k) δ-1k)))
			  oj (d- 1.0d0 oj)))
	(finally (return δ))))

@export
(defun generate-all-δ (w o z0)
  ;; o :: すべての出力層, inputは含まず. 3こ
  ;; w :: すべての重み 2こ
  ;; z0 :: 一番後の教師信号
  (iter
	(with δ = (make-array (length o)));; δの数は 3
	(with mostout = (mostout-δ (aref o (1- (length o))) z0))
	(initially
	 (setf (aref δ (1- (length o))) mostout)) ;; n=2につっこんだ
	
	;; (assert (= (1- (length w)) (length o) (length δ)))

	;; n = 1からはじめて下がる
	(for on in-vector o with-index n from (- (length o) 2) downto 0)
	(for wn in-vector w downto 0)
	
	(for δn-1 previous δn)
	(for δn
		 first (hidden-δ on wn mostout)
		 then  (hidden-δ on wn δn-1))
	(setf (aref δ n) δn)
	(finally (return δ))))

@export
(defun back-propagate (x z0 nn)
  (with-slots (w nodes) nn
	(iter
	  (with o = (layer-status-after-propagation x nn));; 出力の数は3
	  (with δ = (generate-all-δ w o z0))

	  (for δn   in-vector δ from 1);; δの数は 3
	  (for wn   in-vector w from 0);; 重みの数は2
	  (for on-1 in-vector o from 0);; 出力の数は3 
	  ;; (assert (= (1+ (length w)) (length o) (length δ)))
	  (for (im jm) = (array-dimensions wn))
	  ;; (break "~@{~a ~}" (length w) (length o) (length δ))
	  ;; (break "~@{~a ~}" im jm (array-dimensions wn) (length on-1) (length δn))
	  ;; (assert (= (1- im) (length on-1))
	  ;; 		  nil "~@{~a ~}" '(= im (length on-1)) im (length on-1))
	  ;; (assert (= jm (length δn))
	  ;; 		  nil "~@{~a ~}" '(= jm (length δn)) jm (length δn))
	  (iter
		(for i below (1- im))
		(iter
		  (for j below jm)
		  (incf (aref wn i j)
				(d* +η+ (aref δn j) (aref on-1 i)))))
	  (iter
	  	(for j below jm)
	  	(incf (aref wn (1- im) j)
	  		  (d* +η+ (aref δn j) 1.0d0)))
	  )))

@export
@doc "a utility function which creates correct input for BP-TEACH.
all values should be of type =double-float="
(defun make-input (&rest args)
  (coerce args '(array *desired-type* 1)))

@export
@doc "
FN : function

utility function which apply its arguments to FN and returns formatted
output for BP-TEACH. all values should be of type =double-float=."
(defun make-output (fn &rest args)
  (multiple-value-list (apply fn args)))

@export
(defun make-output-from-input (fn input)
  (apply #'make-output fn (coerce input 'list)))

(defun read-new-value ()
  (format t "Enter a new value: ")
  (multiple-value-list (eval (read))))

@export
@doc "
bp-teach (fn, nodes, &key iteration, nn) -> nn

FN : the target function. ( input-arguments* -> output-arguments* )

ITERATION : a =fixnum=

NODES : ({ number-of-nodes-in-layer }*)

NUMBER-OF-NODES-IN-LAYER : a =fixnum=

NN : an instance of =neural-network=

let I = [0.0d0,1.0d0] .
function FN should accept n =double-float= arguments within I
and is expected to return m =double-float= values also within I.
n should match the first =fixnum= in the NODES , and m should
match the last =fixnum= in the NODES.

if NN is unspecified, NODES argument is used to create 
a new instance of =neural-network=
otherwise, NODES will be ignored and it will
conduct further teaching on NN.

ITERATION determines iteration number of
 back-propagation algorhithm, defaulted to 10000.

restarts:

+ /Restart/ add-iteration (new-i)

"
(defun bp-teach (fn nodes
				 &key
				 (iteration 10000)
				 nn)
  (let ((nn (or nn
				(make-instance
				 'neural-network
				 :nodes (coerce nodes 'vector))))
		(dim (car nodes)))
	(print nn)
	(iter 
	  (summing iteration into total)
	  (iter (repeat iteration)
			(with input = (make-array dim :element-type '*desired-type*))
			(iter (for xi in-vector input with-index i)
				  (setf (aref input i) (drandom 1.0d0)))
			(for z0 = (make-output-from-input fn input))
			(back-propagate input z0 nn))
	  (for j =
		   (/ (iter (repeat 100)
					(with input = (make-array dim :element-type
											  '*desired-type*))
					(iter (for xi in-vector input with-index i)
						  (setf (aref input i) (drandom 1.0d0)))
					(for z0 = (make-output-from-input fn input))
					(summing (j-at input z0 nn)))
			  100))
	  (format t "~%~ath bp: average J=~6f" total j)
	  (restart-case
		  (error "iteration finished. what do you do?")
		(add-iteration (new-i)
		  :interactive read-new-value
		  (setf iteration new-i))
		(finish-learning ()
		  (return nn))))))

