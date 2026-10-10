"PharoTaskMiner benchmark harness. Run headless, never saved:
     PTM_JOB=/path/job.json ./pharo PTMBench.image st harness.st
 Reads a job (JSON), installs the task's classes in this image, runs the checks the job
 asks for and writes one JSON result to job.out. See docs/4-benchmark.md."
| job out result classes methods byClass compiled problems defineClass compileMethod testClass testSelectors
  runTest runAll targets stubSource installed restore json timeLimit repeat known analysis facts ston |
job := STONJSON fromString: (Smalltalk os environment at: 'PTM_JOB') asFileReference contents.
out := (job at: 'out') asFileReference.
result := Dictionary new.
problems := OrderedCollection new.
json := [ result at: 'problems' put: problems asArray. out ensureDelete. out writeStreamDo: [ :s | s nextPutAll: (STONJSON toStringPretty: result) ] ].
result at: 'image' put: SystemVersion current asString; at: 'vm' put: Smalltalk vm version asString.
[
classes := job at: 'classes'.
methods := job at: 'methods'.
timeLimit := (job at: 'test_time_limit_seconds' ifAbsent: [ 10 ]) seconds.
repeat := job at: 'repeat' ifAbsent: [ 3 ].

"--- 1. preflight: names free, superclasses known, every method parses -----------------"
known := (classes collect: [ :c | c at: 'name' ]) asSet.
classes do: [ :c |
	(Smalltalk globals includesKey: (c at: 'name') asSymbol) ifTrue: [
		problems add: { 'kind' -> 'class_name_conflict'. 'detail' -> (c at: 'name') } asDictionary ].
	((known includes: (c at: 'superclass')) or: [ Smalltalk globals includesKey: (c at: 'superclass') asSymbol ]) ifFalse: [
		problems add: { 'kind' -> 'unknown_superclass'. 'detail' -> (c at: 'superclass') } asDictionary ] ].
methods do: [ :m |
	[ m at: 'selector' put: (RBParser parseMethod: (m at: 'source')) selector asString ]
		on: Error do: [ :e |
			problems add: { 'kind' -> 'syntax_error'. 'class' -> (m at: 'class'). 'detail' -> e messageText asString.
				'source' -> (m at: 'source') } asDictionary ] ].
(job at: 'candidates' ifAbsent: [ #() ]) do: [ :cand | (cand at: 'methods') do: [ :m |
	[ m at: 'selector' put: (RBParser parseMethod: (m at: 'source')) selector asString ]
		on: Error do: [ :e | m at: 'syntax_error' put: e messageText asString ] ] ].
problems ifNotEmpty: [ result at: 'stage' put: 'preflight'. json value. ^ Smalltalk exitSuccess ].

"--- 2. install ------------------------------------------------------------------------"
defineClass := [ :c | | builder |
	builder := (Smalltalk globals at: (c at: 'superclass') asSymbol) << (c at: 'name') asSymbol.
	builder slots: ((c at: 'instance_variables' ifAbsent: [ #() ]) collect: [ :n | n asSymbol ]) asArray.
	builder sharedVariables: ((c at: 'class_variables' ifAbsent: [ #() ]) collect: [ :n | n asSymbol ]) asArray.
	builder package: (job at: 'package').
	builder install.
	(Smalltalk globals at: (c at: 'name') asSymbol) comment: (c at: 'comment' ifAbsent: [ '' ]) ].
"superclasses first"
[ | pending progress |
	pending := classes asOrderedCollection.
	[ pending isEmpty ] whileFalse: [
		progress := pending select: [ :c | Smalltalk globals includesKey: (c at: 'superclass') asSymbol ].
		progress ifEmpty: [ self error: 'cyclic class hierarchy' ].
		progress do: [ :c | defineClass value: c ].
		pending removeAll: progress ] ] value.
compileMethod := [ :m | | cls |
	cls := Smalltalk globals at: (m at: 'class') asSymbol.
	(m at: 'side' ifAbsent: [ 'instance' ]) = 'class' ifTrue: [ cls := cls class ].
	[ cls compile: (m at: 'source') classified: (m at: 'protocol' ifAbsent: [ 'benchmark' ]) ]
		on: Warning do: [ :w | w resume: nil ].
	cls >> (m at: 'selector') asSymbol ].
compiled := OrderedCollection new.
methods do: [ :m |
	[ compiled add: m -> (compileMethod value: m) ]
		on: Error do: [ :e |
			problems add: { 'kind' -> 'compile_error'. 'class' -> (m at: 'class'). 'selector' -> (m at: 'selector').
				'detail' -> e description asString } asDictionary ] ].

"--- 3. static facts: undeclared names, selectors nobody implements, what the code touches"
facts := OrderedCollection new.
compiled do: [ :assoc | | cm ast undeclared sends unimplemented globals |
	cm := assoc value.
	ast := cm ast.
	undeclared := (ast allChildren select: [ :n | n isVariable and: [ n variable isUndeclaredVariable ] ])
		collect: [ :n | n name asString ].
	sends := (ast allChildren select: [ :n | n isMessage ]) collect: [ :n | n selector asString ].
	unimplemented := sends asSet select: [ :s | (SystemNavigation default allImplementorsOf: s asSymbol) isEmpty ].
	globals := (ast allChildren select: [ :n | n isVariable and: [ n variable isGlobalVariable ] ])
		collect: [ :n | n name asString ].
	undeclared asSet do: [ :n |
		problems add: { 'kind' -> 'undeclared_variable'. 'class' -> (assoc key at: 'class').
			'selector' -> (assoc key at: 'selector'). 'detail' -> n } asDictionary ].
	unimplemented do: [ :s |
		problems add: { 'kind' -> 'unimplemented_selector'. 'class' -> (assoc key at: 'class').
			'selector' -> (assoc key at: 'selector'). 'detail' -> s } asDictionary ].
	facts add: { 'class' -> (assoc key at: 'class'). 'selector' -> (assoc key at: 'selector').
		'role' -> (assoc key at: 'role' ifAbsent: [ 'context' ]).
		'sends' -> sends asSet asSortedCollection asArray. 'globals' -> globals asSet asSortedCollection asArray } asDictionary ].
result at: 'facts' put: facts asArray.
problems ifNotEmpty: [ result at: 'stage' put: 'install'. json value. ^ Smalltalk exitSuccess ].

"--- 4. the tests on the reference solution: repeated, and in reverse order -------------"
testClass := Smalltalk globals at: (job at: 'test_class') asSymbol.
testSelectors := (testClass selectors select: [ :s | s beginsWith: 'test' ]) asSortedCollection asArray.
runTest := [ :sel |
	[ [ [ (testClass selector: sel) runCase. 'pass' -> '' ]
			on: Warning do: [ :w | w resume: nil ] ]
		on: Error, TestFailure, Halt do: [ :e |
			((e isKindOf: TestFailure) ifTrue: [ 'failure' ] ifFalse: [ 'error' ]) -> (e description asString first: (e description size min: 300)) ] ]
		valueWithin: timeLimit onTimeout: [ 'timeout' -> '' ] ].
runAll := [ :selectors | | d |
	d := OrderedDictionary new.
	selectors do: [ :sel | | r | r := runTest value: sel. d at: sel asString put: { 'outcome' -> r key. 'detail' -> r value } asDictionary ].
	d ].
result at: 'tests' put: (testSelectors collect: [ :s | s asString ]).
[ | runs green |
	runs := OrderedCollection new.
	green := true.
	1 to: repeat do: [ :i |
		green ifTrue: [ | r |
			r := runAll value: (i even ifTrue: [ testSelectors reversed ] ifFalse: [ testSelectors ]).
			runs add: r.
			green := r allSatisfy: [ :each | (each at: 'outcome') = 'pass' ] ] ].
	result at: 'reference_runs' put: runs asArray.
	"nothing below means anything unless the reference solution passes its own tests"
	(green and: [ testSelectors notEmpty ]) ifFalse: [ result at: 'stage' put: 'done'. json value. ^ Smalltalk exitSuccess ] ] value.

"--- 5. fail-to-pass: each target method replaced by a stub -----------------------------"
targets := compiled select: [ :assoc | (assoc key at: 'role' ifAbsent: [ '' ]) = 'target' ].
stubSource := [ :cm | | ast |
	ast := cm ast.
	String streamContents: [ :s |
		ast arguments isEmpty
			ifTrue: [ s nextPutAll: ast selector ]
			ifFalse: [ ast selector keywords with: ast arguments do: [ :k :a | s nextPutAll: k; space; nextPutAll: a name; space ] ].
		s cr; tab; nextPutAll: '^ self error: ''not implemented''' ] ].
restore := [ targets do: [ :assoc | compileMethod value: assoc key ] ].
result at: 'stub_runs' put: (targets collect: [ :assoc | | r |
	assoc value methodClass compile: (stubSource value: assoc value) classified: 'benchmark'.
	r := runAll value: testSelectors.
	compileMethod value: assoc key.
	{ 'class' -> (assoc key at: 'class'). 'selector' -> (assoc key at: 'selector'). 'tests' -> r } asDictionary ]) asArray.
result at: 'all_stubbed_run' put: ([ | r |
	targets do: [ :assoc | assoc value methodClass compile: (stubSource value: assoc value) classified: 'benchmark' ].
	r := runAll value: testSelectors.
	restore value.
	r ] value).

"--- 6. candidate solutions (independent solvers, or a model under evaluation) -----------"
result at: 'candidates' put: ((job at: 'candidates' ifAbsent: [ #() ]) collect: [ :cand | | errors added r |
	errors := OrderedCollection new.
	added := OrderedCollection new.
	"the candidate starts from the stubs, exactly like a model that is given the task"
	targets do: [ :assoc | assoc value methodClass compile: (stubSource value: assoc value) classified: 'benchmark' ].
	(cand at: 'methods') do: [ :m |
		(m includesKey: 'syntax_error')
			ifTrue: [ errors add: 'syntax error: ', (m at: 'syntax_error') ]
			ifFalse: [
				[ | cm existed cls |
					cls := Smalltalk globals at: (m at: 'class') asSymbol.
					(m at: 'side' ifAbsent: [ 'instance' ]) = 'class' ifTrue: [ cls := cls class ].
					existed := cls includesSelector: (m at: 'selector') asSymbol.
					cm := compileMethod value: m.
					existed ifFalse: [ added add: cm ].
					(cm ast allChildren select: [ :n | n isVariable and: [ n variable isUndeclaredVariable ] ])
						do: [ :n | errors add: 'undeclared variable: ', n name ] ]
					on: Error do: [ :e | errors add: 'compile error: ', e description asString ] ] ].
	r := errors isEmpty ifTrue: [ runAll value: testSelectors ] ifFalse: [ OrderedDictionary new ].
	added do: [ :cm | cm methodClass removeSelector: cm selector ].
	restore value.
	{ 'name' -> (cand at: 'name'). 'errors' -> errors asArray. 'tests' -> r } asDictionary ]) asArray.

"--- 7. mutation analysis of the target methods with MuTalk ------------------------------"
((job at: 'mutation' ifAbsent: [ false ]) and: [ targets notEmpty ]) ifTrue: [
	[ | general |
		analysis := (Smalltalk globals at: #MTAnalysis) new.
		analysis classesToMutate: (targets collect: [ :a | a value methodClass instanceSide ]) asSet asArray.
		"the methods as they are installed now (they were recompiled by the steps above)"
		analysis methodsToMutate: (targets collect: [ :a | a value methodClass >> a value selector ]) asArray.
		analysis testClasses: { testClass }.
		analysis testSelectionStrategy: (Smalltalk globals at: #MTAllTestsMethodsRunningTestSelectionStrategy) new.
		analysis budget: (Smalltalk globals at: #MTFreeBudget) new.
		analysis logger: (Smalltalk globals at: #MTNullLogger) new.
		analysis run.
		general := analysis generalResult.
		result at: 'mutation' put: {
			'tool' -> 'MuTalk'.
			'evaluated' -> general numberOfEvaluatedMutants.
			'killed' -> general numberOfKilledMutants.
			'terminated' -> general numberOfTerminatedMutants.
			'alive' -> general numberOfAliveMutants.
			'mutants' -> (general particularResults collect: [ :r | {
				'operator' -> r mutant operator class name asString.
				'selector' -> r mutant originalMethod selector asString.
				'class' -> r mutant originalClass name asString.
				'status' -> (r killed ifTrue: [ 'killed' ] ifFalse: [ r terminated ifTrue: [ 'terminated' ] ifFalse: [ 'survived' ] ]).
				'source' -> r mutant modifiedSource asString } asDictionary ]) asArray } asDictionary ]
		on: Error do: [ :e | result at: 'mutation' put: { 'tool' -> 'MuTalk'. 'error' -> e description asString } asDictionary ].
	restore value.
	"the analysis must leave the reference in place: run the tests once more"
	result at: 'after_mutation_run' put: (runAll value: testSelectors) ].

"--- 8. export the task as a Tonel package ----------------------------------------------"
(job at: 'export_dir' ifAbsent: [ nil ]) ifNotNil: [ :dir | | ref |
	ref := dir asFileReference ensureCreateDirectory.
	[ (TonelWriter on: ref) writeSnapshot: (MCPackage named: (job at: 'package')) snapshot.
		result at: 'exported' put: true ]
		on: Error do: [ :e | result at: 'exported' put: false; at: 'export_error' put: e description asString ] ].
result at: 'stage' put: 'done'.
] on: Error do: [ :e |
	result at: 'stage' put: 'harness_error'; at: 'harness_error' put: e description asString;
		at: 'stack' put: ((e signalerContext stack first: (e signalerContext stack size min: 12)) collect: [ :c | c printString ]) asArray ].
json value.
Smalltalk exitSuccess
