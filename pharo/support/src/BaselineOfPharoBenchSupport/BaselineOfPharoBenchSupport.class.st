Class {
	#name : #BaselineOfPharoBenchSupport,
	#superclass : #BaselineOf,
	#category : #BaselineOfPharoBenchSupport
}

{ #category : #baselines }
BaselineOfPharoBenchSupport >> baseline: spec [
	<baseline>
	spec for: #common do: [ spec package: 'PharoBench-Support' ]
]
