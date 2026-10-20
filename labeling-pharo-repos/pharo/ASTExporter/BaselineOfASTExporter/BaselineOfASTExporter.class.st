Class {
	#name : 'BaselineOfASTExporter',
	#superclass : 'BaselineOf',
	#category : 'BaselineOfASTExporter',
	#package : 'BaselineOfASTExporter'
}

{ #category : 'baselines' }
BaselineOfASTExporter >> baseline: spec [

	<baseline>
	spec for: #common do: [ spec package: 'ASTExporter' ]
]
