
(in-package :cl-user)

(defpackage fri34.report1
  (:use :cl
		:alexandria
		:mon4.utilities
		:cl-test-more
		:iterate
		:annot
		:annot.doc
		:vecto)
  (:shadow :terminate :rotate))

(in-package :fri34.report1)
(annot:enable-annot-syntax)

(declaim (optimize (debug 3)))

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

(defun sigmoid (gain)
  @type *desired-type* gain
  (lambda (x)
	@type *desired-type* x
	(d/ (d+ 1.0d0 (dexp (d- (d* gain x)))))))

(setf (symbol-function 'sigmoid1) (sigmoid 0.5d0))

(defun dsigmoid (gain)
  @type *desired-type* gain
  (lambda (x)
	@type *desired-type* x
	(let ((sigx (funcall (sigmoid gain) x)))
	  @type *desired-type* sigx
	  (d* gain sigx (d- 1.0d0 sigx)))))

(setf (symbol-function 'dsigmoid1) (dsigmoid 1.0d0))

(defun randomize (ary limit)
  (iter (for i from 0 below (array-total-size ary))
		(setf (row-major-aref ary i)
			  (d- (drandom (d* 2.0d0 limit)) limit)))
  ary)

(defclass neural-network ()
  ((input-nodes :type fixnum :initform 2 :initarg :input)
   (intermediate-nodes :type fixnum :initform 10 :initarg :intermediate)
   (output-nodes :type fixnum :initform 1 :initarg :output)
   (w1 :type (array double-float (* *))
	   :accessor input-intermediate-weight :accessor w1-of)
   (w2 :type (array double-float (* *))
	   :accessor intermediate-output-weight :accessor w2-of)))

(defun make-weight (n1 n2)
  (make-array (list (1+ n1) n2) :element-type '*desired-type*))

(defmethod initialize-instance :after ((nn neural-network) &rest args)
  @ignore args
  (with-slots (w1 w2 input-nodes intermediate-nodes output-nodes) nn
	(setf w1
		  (randomize
		   (make-weight input-nodes intermediate-nodes) 1.0d-1)
		  w2
		  (randomize
		   (make-weight intermediate-nodes output-nodes) 1.0d-1))))


(defun layer-input (l1 w)
  (iter 
	(with n1 = (1- (array-dimension w 0)))
	(with n2 = (array-dimension w 1))
	(with l2 = (make-array n2 :element-type '*desired-type*))
	(for j below n2)
	(setf (aref l2 j)
		  (d+ (iter (for i below n1)
					(summing (d* (aref w i j)
								 (aref l1 i))))
			  (aref w n1 j))) 			; * 1.0d0
	(finally (return l2))))

(defun layer-output (i)
  (map 'vector #'sigmoid1 i))

;; returns vector
(defun propagate (input nn)
  (with-slots (w1 w2) nn
	(let* ((u (layer-input input w1))
		   (y (layer-output u))
		   (v (layer-input y w2))
		   (z (layer-output v)))
	  (values u y v z))))

(defun sqdiff (a b)
  (d^2 (d- a b)))

(defun j-at (x z0 nn &aux
			 (z? (multiple-value-bind (u y v z)
					 (propagate x nn)
				   @ignore u
				   @ignore y
				   @ignore v
				   z)))
  (reduce #'+ (map 'vector #'sqdiff z? z0)))

@doc "学習率パラメータ"
(defparameter +η+ 8.0d-2)

(defun back-propagate (x z0 nn)
  (with-slots (w1 w2 input-nodes output-nodes
				  intermediate-nodes) nn
	(multiple-value-bind (u y v z) (propagate x nn)
	  (let ((w1-next (copy-array w1))
			(w2-next (copy-array w2)))
		(flet ((mostout-mapfn (j yj)
				 (iter (for k from 0 below output-nodes)
					   (for zk in-vector z)
					   ;;(for vk in-vector v)
					   (for tk in z0)
					   (incf (aref w2-next j k)
							 (d* +η+ (d- tk zk) zk (d- 1.0d0 zk) yj))))
			   (hidden-mapfn (i xi)
				 (iter (for j from 0 below intermediate-nodes)
					   (for yj in-vector y)
					   ;;(for uj in-vector u)
					   (incf (aref w1-next i j)
							 (d* +η+
								 (iter (for k from 0 below output-nodes)
									   (for zk in-vector z)
									   ;;(for vk in-vector v)
									   (for tk in z0)
									   (summing
										(d* (aref w2 j k)
											(d- tk zk) zk (d- 1.0d0 zk))))
								 yj (d- 1.0d0 yj) xi)))))
		  (iter (for j from 0 below intermediate-nodes)
				(for yj in-vector y)
				(mostout-mapfn j yj))
		  (mostout-mapfn intermediate-nodes 1.0d0)
		  (iter (for i from 0 below input-nodes)
				(for xi in-vector x)
				(hidden-mapfn i xi))
		  (hidden-mapfn input-nodes 1.0d0))
		(setf w1 w1-next w2 w2-next)
		(values nn z)))))

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

(defun make-input (&rest args)
  (coerce args 'vector))
(defun make-output (fn &rest args)
  (multiple-value-list (apply fn args)))

(defun bp-test (fn &optional (path-bp "bp.png") (path-ans "bp-answer.png"))
  (let* ((nn (make-instance 'neural-network :input 2 :intermediate 10)))
	(iter (with print-par = 200000)
		  (for i from 0 to 10)
		  (for x0 = (drandom 1.0d0))
		  (for y0 = (drandom 1.0d0))
		  (for x = (make-input x0 y0))
		  (for z0 = (make-output fn x0 y0))
		  (for before = (j-at x z0 nn))
		  (back-propagate x z0 nn)
		  (for after = (j-at x z0 nn))
		  (format t
				  "~%~ath bp:f(~3f, ~3f)=~{~3f~} true:~{~3f~} J=~6f dJ=~6f"
				  (* i print-par) x0 y0
				  (multiple-value-bind (u y v z)
					  (propagate (make-input x0 y0) nn)
					@ignore u y v
					(coerce z 'list))
				  z0 after (d- after before))
		  (iter (repeat (1- print-par))
				(for x0 = (drandom 1.0d0))
				(for y0 = (drandom 1.0d0))
				(for x = (make-input x0 y0))
				(for z0 = (make-output fn x0 y0))
				(back-propagate x z0 nn)))
	(with-slots (w1 w2) nn
	  (format t "~%w1 and w2 after the learning is: ~%~A~%~A" w1 w2))
	(format t "~%drawing the learned function...")
	(write-func-to-png
	 (lambda (x0 y0)
	   (multiple-value-bind (u y v z)
		   (propagate (make-input x0 y0) nn)
		 @ignore u y v
		 (aref z 0))) path-bp)
	(format t "~%drawing the original function...")
	(write-func-to-png fn path-ans)))
  
