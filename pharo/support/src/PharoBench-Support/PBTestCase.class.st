"
I am the superclass of every PharoBench test class.

`task` holds a fresh instance of the class under test (set in each test class's `setUp`).

`assertResult:equals:` and `assertResult:equals:tolerance:` compare an answer with an expected
value the way EvalPlus compares Python results, so that tests compiled from HumanEval+/MBPP+
keep their meaning: exact equality, except for floats (absolute tolerance, at least 1e-6 when the
expected value holds floats, plus a relative tolerance of 1e-7); a Python list or tuple is any
sequenceable collection; strings and symbols compare by characters; dictionaries by keys and values;
sets by membership.
"
Class {
	#name : #PBTestCase,
	#superclass : #TestCase,
	#instVars : [
		'task'
	],
	#category : #'PharoBench-Support'
}

{ #category : #testing }
PBTestCase class >> isAbstract [
	^ self == PBTestCase
]

{ #category : #asserting }
PBTestCase >> assertResult: actual equals: expected [
	self assertResult: actual equals: expected tolerance: 0
]

{ #category : #asserting }
PBTestCase >> assertResult: actual equals: expected tolerance: atol [
	| tol |
	tol := (atol = 0 and: [ self holdsFloats: expected ])
		ifTrue: [ 1.0e-6 ]
		ifFalse: [ atol ].
	self
		assert: (self value: actual matches: expected tolerance: tol)
		description: [ 'Expected ' , (self shortPrint: expected) , ' but got ' , (self shortPrint: actual) ]
]

{ #category : #private }
PBTestCase >> holdsFloats: anObject [
	anObject isFloat ifTrue: [ ^ true ].
	(anObject isCollection and: [ anObject isString not and: [ anObject isDictionary not ] ])
		ifTrue: [ ^ anObject notEmpty and: [ anObject allSatisfy: [ :each | each isFloat ] ] ].
	^ false
]

{ #category : #private }
PBTestCase >> shortPrint: anObject [
	| s |
	s := anObject printString.
	^ s size > 300
		ifTrue: [ (s first: 300) , '...' ]
		ifFalse: [ s ]
]

{ #category : #private }
PBTestCase >> value: actual matches: expected tolerance: tol [
	(expected isNumber and: [ actual isNumber ]) ifTrue: [
		(expected isFloat or: [ actual isFloat or: [ tol > 0 ] ]) ifFalse: [ ^ actual = expected ].
		(expected isFloat and: [ expected isNaN ]) ifTrue: [ ^ actual isFloat and: [ actual isNaN ] ].
		(expected isFloat and: [ expected isInfinite ]) ifTrue: [ ^ actual = expected ].
		^ (actual - expected) abs <= (tol + (1.0e-7 * expected abs)) ].
	(expected isString or: [ expected isSymbol ]) ifTrue: [
		^ (actual isString or: [ actual isSymbol ]) and: [ actual asString = expected asString ] ].
	expected isDictionary ifTrue: [
		actual isDictionary ifFalse: [ ^ false ].
		actual size = expected size ifFalse: [ ^ false ].
		^ expected keys allSatisfy: [ :k |
			(actual includesKey: k) and: [ self value: (actual at: k) matches: (expected at: k) tolerance: tol ] ] ].
	(expected isKindOf: Set) ifTrue: [
		(actual isCollection and: [ actual isString not ]) ifFalse: [ ^ false ].
		actual asSet size = expected size ifFalse: [ ^ false ].
		^ expected allSatisfy: [ :e | actual anySatisfy: [ :a | self value: a matches: e tolerance: tol ] ] ].
	(expected isCollection and: [ expected isSequenceable ]) ifTrue: [
		(actual isCollection and: [ actual isSequenceable and: [ actual isString not ] ]) ifFalse: [ ^ false ].
		actual size = expected size ifFalse: [ ^ false ].
		^ (1 to: expected size) allSatisfy: [ :i |
			self value: (actual at: i) matches: (expected at: i) tolerance: tol ] ].
	^ actual = expected
]
