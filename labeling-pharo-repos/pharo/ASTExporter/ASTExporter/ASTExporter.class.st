"
I parse method sources with Pharo's RBParser and export a JSON AST using the
same node vocabulary as the Python parser of PharoTaskMiner
(Method, Sequence, Message, Cascade, CascadeReceiver, Assignment, Return,
Block, Literal, DynamicArray, Variable, Pragma).

Input: JSON Lines with {""id"": ..., ""source"": ...}. Output: JSON Lines with
{""id"": ..., ""ast"": ...} or {""id"": ..., ""error"": ...}.
Nothing is compiled or executed; sources are only parsed.

	ASTExporter exportFrom: '/tmp/methods.jsonl' to: '/tmp/asts.jsonl'
"
Class {
	#name : 'ASTExporter',
	#superclass : 'Object',
	#instVars : [
		'source'
	],
	#category : 'ASTExporter',
	#package : 'ASTExporter'
}

{ #category : 'exporting' }
ASTExporter class >> exportFrom: inputPath to: outputPath [

	| out |
	out := outputPath asFileReference.
	out ensureDelete.
	out writeStreamDo: [ :w |
		inputPath asFileReference readStreamDo: [ :r |
			[ r atEnd ] whileFalse: [
				| line record result |
				line := r nextLine.
				line isEmptyOrNil ifFalse: [
					record := STONJSON fromString: line.
					result := self new exportRecord: record.
					w nextPutAll: (STONJSON toString: result); lf ] ] ] ]
]

{ #category : 'exporting' }
ASTExporter >> exportRecord: record [

	| ast |
	source := record at: 'source'.
	ast := RBParser
		       parseMethod: source
		       onError: [ :msg :pos |
			       ^ Dictionary new
				         at: 'id' put: (record at: 'id');
				         at: 'error' put: msg asString , ' (offset ' , (pos - 1) printString , ')';
				         yourself ].
	^ Dictionary new
		  at: 'id' put: (record at: 'id');
		  at: 'ast' put: (self visit: ast);
		  yourself
]

{ #category : 'private' }
ASTExporter >> lineOf: aNode [

	| start |
	start := aNode start ifNil: [ ^ 1 ].
	^ ((source first: ((start - 1) max: 0)) occurrencesOf: Character lf) + 1
]

{ #category : 'private' }
ASTExporter >> literalKind: anObject [

	anObject isSymbol ifTrue: [ ^ 'symbol' ].
	anObject isString ifTrue: [ ^ 'string' ].
	anObject isCharacter ifTrue: [ ^ 'character' ].
	anObject isNumber ifTrue: [ ^ 'number' ].
	anObject isNil ifTrue: [ ^ 'nil' ].
	(anObject == true or: [ anObject == false ]) ifTrue: [ ^ 'boolean' ].
	(anObject isKindOf: ByteArray) ifTrue: [ ^ 'bytearray' ].
	^ 'array'
]

{ #category : 'private' }
ASTExporter >> node: type for: aNode [

	^ Dictionary new
		  at: 'type' put: type;
		  at: 'start' put: (aNode start ifNil: [ 0 ] ifNotNil: [ :s | s - 1 ]);
		  at: 'line' put: (self lineOf: aNode);
		  yourself
]

{ #category : 'visiting' }
ASTExporter >> visit: aNode [

	aNode isMethod ifTrue: [ ^ self visitMethod: aNode ].
	aNode isSequence ifTrue: [ ^ self visitSequence: aNode ].
	aNode isCascade ifTrue: [ ^ self visitCascade: aNode ].
	aNode isMessage ifTrue: [ ^ self visitMessage: aNode receiver: (self visit: aNode receiver) ].
	aNode isAssignment ifTrue: [
		^ (self node: 'Assignment' for: aNode)
			  at: 'variable' put: aNode variable name asString;
			  at: 'value' put: (self visit: aNode value);
			  yourself ].
	aNode isReturn ifTrue: [
		^ (self node: 'Return' for: aNode)
			  at: 'value' put: (self visit: aNode value);
			  yourself ].
	aNode isBlock ifTrue: [
		^ (self node: 'Block' for: aNode)
			  at: 'arguments' put: (aNode arguments collect: [ :a | a name asString ]) asArray;
			  at: 'body' put: (self visitSequence: aNode body);
			  yourself ].
	aNode isDynamicArray ifTrue: [
		^ (self node: 'DynamicArray' for: aNode)
			  at: 'statements' put: (aNode statements collect: [ :s | self visit: s ]) asArray;
			  yourself ].
	aNode isLiteralNode ifTrue: [ ^ self visitLiteral: aNode ].
	aNode isVariable ifTrue: [
		^ (self node: 'Variable' for: aNode)
			  at: 'name' put: aNode name asString;
			  yourself ].
	^ (self node: 'Unsupported' for: aNode)
		  at: 'class' put: aNode class name asString;
		  yourself
]

{ #category : 'visiting' }
ASTExporter >> visitCascade: aNode [

	^ (self node: 'Cascade' for: aNode)
		  at: 'receiver' put: (self visit: aNode receiver);
		  at: 'messages' put: (aNode messages collect: [ :m |
				   self visitMessage: m receiver: (Dictionary new at: 'type' put: 'CascadeReceiver'; yourself) ]) asArray;
		  yourself
]

{ #category : 'visiting' }
ASTExporter >> visitLiteral: aNode [

	| value |
	value := aNode value.
	^ (self node: 'Literal' for: aNode)
		  at: 'kind' put: (self literalKind: value);
		  at: 'value' put: ((value isString or: [ value isCharacter ])
				   ifTrue: [ value asString ]
				   ifFalse: [ aNode sourceText ]);
		  yourself
]

{ #category : 'visiting' }
ASTExporter >> visitMessage: aNode receiver: receiverDictionary [

	| d |
	d := self node: 'Message' for: aNode.
	aNode keywordsPositions ifNotNil: [ :positions |
		positions isEmpty ifFalse: [
			d at: 'start' put: positions first - 1.
			d at: 'line' put: (((source first: ((positions first - 1) max: 0)) occurrencesOf: Character lf) + 1) ] ].
	^ d
		  at: 'receiver' put: receiverDictionary;
		  at: 'selector' put: aNode selector asString;
		  at: 'arguments' put: (aNode arguments collect: [ :a | self visit: a ]) asArray;
		  at: 'kind' put: (aNode isUnary
				   ifTrue: [ 'unary' ]
				   ifFalse: [ aNode isBinary ifTrue: [ 'binary' ] ifFalse: [ 'keyword' ] ]);
		  yourself
]

{ #category : 'visiting' }
ASTExporter >> visitMethod: aNode [

	^ (self node: 'Method' for: aNode)
		  at: 'selector' put: aNode selector asString;
		  at: 'arguments' put: (aNode arguments collect: [ :a | a name asString ]) asArray;
		  at: 'pragmas' put: (aNode pragmas collect: [ :p |
				   (self node: 'Pragma' for: p)
					   at: 'selector' put: p selector asString;
					   at: 'arguments' put: (p arguments collect: [ :a | a sourceText ]) asArray;
					   yourself ]) asArray;
		  at: 'body' put: (self visitSequence: aNode body);
		  yourself
]

{ #category : 'visiting' }
ASTExporter >> visitSequence: aNode [

	^ (self node: 'Sequence' for: aNode)
		  at: 'temporaries' put: (aNode temporaries collect: [ :t | t name asString ]) asArray;
		  at: 'statements' put: (aNode statements collect: [ :s | self visit: s ]) asArray;
		  yourself
]
