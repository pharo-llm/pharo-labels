"PharoTaskMiner: install the whole benchmark the way a user would, then run every test class.
     PTM_JOB=/path/job.json ./pharo PTMBench.image st install.st
 job: out, repository (directory holding the Tonel packages), baseline, test_classes."
| job out result json |
job := STONJSON fromString: (Smalltalk os environment at: 'PTM_JOB') asFileReference contents.
out := (job at: 'out') asFileReference.
result := Dictionary new.
json := [ out ensureDelete. out writeStreamDo: [ :s | s nextPutAll: (STONJSON toStringPretty: result) ] ].
result at: 'image' put: SystemVersion current asString.
[
	[ Metacello new
		baseline: (job at: 'baseline');
		repository: 'tonel://', (job at: 'repository');
		load ] on: Warning do: [ :w | w resume: nil ].
	result at: 'loaded' put: true.
	result at: 'classes' put: ((job at: 'test_classes') collect: [ :name | | d cls |
		d := Dictionary new.
		d at: 'test_class' put: name.
		cls := Smalltalk globals at: name asSymbol ifAbsent: [ nil ].
		cls
			ifNil: [ d at: 'missing' put: true ]
			ifNotNil: [ | r |
				r := [ cls buildSuite run ] on: Warning do: [ :w | w resume: nil ].
				d at: 'run' put: r runCount; at: 'passed' put: r passedCount;
					at: 'failed' put: ((r failures collect: [ :t | t selector asString ]) asArray
						, (r errors collect: [ :t | t selector asString ]) asArray) ].
		d ]) asArray.
	result at: 'stage' put: 'done'
] on: Error do: [ :e |
	result at: 'stage' put: 'install_error'; at: 'detail' put: e description asString ].
json value.
Smalltalk exitSuccess
