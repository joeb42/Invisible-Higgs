??/
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
<
Selu
features"T
activations"T"
Ttype:
2
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.4.12unknown8??,
?
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namelayer_normalization_1/gamma
?
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes	
:?*
dtype0
?
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelayer_normalization_1/beta
?
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes	
:?*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	?*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:?*
dtype0
z
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
??*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:?*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
??*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:?*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	?*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
?
gru_1/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*(
shared_namegru_1/gru_cell_1/kernel
?
+gru_1/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/kernel*
_output_shapes
:		?*
dtype0
?
!gru_1/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!gru_1/gru_cell_1/recurrent_kernel
?
5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_1/gru_cell_1/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
gru_1/gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_namegru_1/gru_cell_1/bias
?
)gru_1/gru_cell_1/bias/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/bias*
_output_shapes
:	?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:?*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:?*
dtype0
?
#Nadam/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Nadam/layer_normalization_1/gamma/m
?
7Nadam/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp#Nadam/layer_normalization_1/gamma/m*
_output_shapes	
:?*
dtype0
?
"Nadam/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Nadam/layer_normalization_1/beta/m
?
6Nadam/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp"Nadam/layer_normalization_1/beta/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameNadam/dense_4/kernel/m
?
*Nadam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_4/kernel/m*
_output_shapes
:	?*
dtype0
?
Nadam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameNadam/dense_4/bias/m
z
(Nadam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_4/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameNadam/dense_5/kernel/m
?
*Nadam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_5/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameNadam/dense_5/bias/m
z
(Nadam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_5/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameNadam/dense_6/kernel/m
?
*Nadam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_6/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameNadam/dense_6/bias/m
z
(Nadam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_6/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameNadam/dense_7/kernel/m
?
*Nadam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_7/kernel/m*
_output_shapes
:	?*
dtype0
?
Nadam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_7/bias/m
y
(Nadam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_7/bias/m*
_output_shapes
:*
dtype0
?
Nadam/gru_1/gru_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*0
shared_name!Nadam/gru_1/gru_cell_1/kernel/m
?
3Nadam/gru_1/gru_cell_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/gru_1/gru_cell_1/kernel/m*
_output_shapes
:		?*
dtype0
?
)Nadam/gru_1/gru_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*:
shared_name+)Nadam/gru_1/gru_cell_1/recurrent_kernel/m
?
=Nadam/gru_1/gru_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp)Nadam/gru_1/gru_cell_1/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/gru_1/gru_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_nameNadam/gru_1/gru_cell_1/bias/m
?
1Nadam/gru_1/gru_cell_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/gru_1/gru_cell_1/bias/m*
_output_shapes
:	?*
dtype0
?
#Nadam/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Nadam/layer_normalization_1/gamma/v
?
7Nadam/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp#Nadam/layer_normalization_1/gamma/v*
_output_shapes	
:?*
dtype0
?
"Nadam/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Nadam/layer_normalization_1/beta/v
?
6Nadam/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp"Nadam/layer_normalization_1/beta/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameNadam/dense_4/kernel/v
?
*Nadam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_4/kernel/v*
_output_shapes
:	?*
dtype0
?
Nadam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameNadam/dense_4/bias/v
z
(Nadam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_4/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameNadam/dense_5/kernel/v
?
*Nadam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_5/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameNadam/dense_5/bias/v
z
(Nadam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_5/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameNadam/dense_6/kernel/v
?
*Nadam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_6/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameNadam/dense_6/bias/v
z
(Nadam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_6/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameNadam/dense_7/kernel/v
?
*Nadam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_7/kernel/v*
_output_shapes
:	?*
dtype0
?
Nadam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_7/bias/v
y
(Nadam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_7/bias/v*
_output_shapes
:*
dtype0
?
Nadam/gru_1/gru_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*0
shared_name!Nadam/gru_1/gru_cell_1/kernel/v
?
3Nadam/gru_1/gru_cell_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/gru_1/gru_cell_1/kernel/v*
_output_shapes
:		?*
dtype0
?
)Nadam/gru_1/gru_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*:
shared_name+)Nadam/gru_1/gru_cell_1/recurrent_kernel/v
?
=Nadam/gru_1/gru_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp)Nadam/gru_1/gru_cell_1/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/gru_1/gru_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_nameNadam/gru_1/gru_cell_1/bias/v
?
1Nadam/gru_1/gru_cell_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/gru_1/gru_cell_1/bias/v*
_output_shapes
:	?*
dtype0

NoOpNoOp
?L
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?L
value?LB?L B?L
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
 
q
axis
	gamma
beta
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
R
#trainable_variables
$	variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
?
9iter

:beta_1

;beta_2
	<decay
=learning_rate
>momentum_cachem?m?m?m?'m?(m?-m?.m?3m?4m??m?@m?Am?v?v?v?v?'v?(v?-v?.v?3v?4v??v?@v?Av?
^
?0
@1
A2
3
4
5
6
'7
(8
-9
.10
311
412
^
?0
@1
A2
3
4
5
6
'7
(8
-9
.10
311
412
 
?
Blayer_metrics
trainable_variables
	variables
Cnon_trainable_variables
Dlayer_regularization_losses

Elayers
Fmetrics
regularization_losses
 
~

?kernel
@recurrent_kernel
Abias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
 

?0
@1
A2

?0
@1
A2
 
?

Kstates
Llayer_metrics
trainable_variables
Mnon_trainable_variables
Nlayer_regularization_losses
	variables

Olayers
Pmetrics
regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Qlayer_metrics
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
	variables

Tlayers
Umetrics
regularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Vlayer_metrics
trainable_variables
Wlayer_regularization_losses
Xnon_trainable_variables
 	variables

Ylayers
Zmetrics
!regularization_losses
 
 
 
?
[layer_metrics
#trainable_variables
\layer_regularization_losses
]non_trainable_variables
$	variables

^layers
_metrics
%regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
`layer_metrics
)trainable_variables
alayer_regularization_losses
bnon_trainable_variables
*	variables

clayers
dmetrics
+regularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
?
elayer_metrics
/trainable_variables
flayer_regularization_losses
gnon_trainable_variables
0	variables

hlayers
imetrics
1regularization_losses
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
?
jlayer_metrics
5trainable_variables
klayer_regularization_losses
lnon_trainable_variables
6	variables

mlayers
nmetrics
7regularization_losses
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_1/gru_cell_1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!gru_1/gru_cell_1/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEgru_1/gru_cell_1/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
?
0
1
2
3
4
5
6
7
	8

o0
p1

?0
@1
A2

?0
@1
A2
 
?
qlayer_metrics
Gtrainable_variables
rlayer_regularization_losses
snon_trainable_variables
H	variables

tlayers
umetrics
Iregularization_losses
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	vtotal
	wcount
x	variables
y	keras_api
p
ztrue_positives
{true_negatives
|false_positives
}false_negatives
~	variables
	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

v0
w1

x	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

z0
{1
|2
}3

~	variables
??
VARIABLE_VALUE#Nadam/layer_normalization_1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Nadam/layer_normalization_1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_5/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_5/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_6/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_6/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_7/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_7/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUENadam/gru_1/gru_cell_1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Nadam/gru_1/gru_cell_1/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/gru_1/gru_cell_1/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Nadam/layer_normalization_1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Nadam/layer_normalization_1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_5/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_5/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_6/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_6/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_7/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_7/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUENadam/gru_1/gru_cell_1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Nadam/gru_1/gru_cell_1/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/gru_1/gru_cell_1/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_3Placeholder*+
_output_shapes
:?????????	*
dtype0* 
shape:?????????	
z
serving_default_input_4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_4gru_1/gru_cell_1/biasgru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kernellayer_normalization_1/gammalayer_normalization_1/betadense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*5J 8? *-
f(R&
$__inference_signature_wrapper_223409
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOp+gru_1/gru_cell_1/kernel/Read/ReadVariableOp5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOp)gru_1/gru_cell_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp7Nadam/layer_normalization_1/gamma/m/Read/ReadVariableOp6Nadam/layer_normalization_1/beta/m/Read/ReadVariableOp*Nadam/dense_4/kernel/m/Read/ReadVariableOp(Nadam/dense_4/bias/m/Read/ReadVariableOp*Nadam/dense_5/kernel/m/Read/ReadVariableOp(Nadam/dense_5/bias/m/Read/ReadVariableOp*Nadam/dense_6/kernel/m/Read/ReadVariableOp(Nadam/dense_6/bias/m/Read/ReadVariableOp*Nadam/dense_7/kernel/m/Read/ReadVariableOp(Nadam/dense_7/bias/m/Read/ReadVariableOp3Nadam/gru_1/gru_cell_1/kernel/m/Read/ReadVariableOp=Nadam/gru_1/gru_cell_1/recurrent_kernel/m/Read/ReadVariableOp1Nadam/gru_1/gru_cell_1/bias/m/Read/ReadVariableOp7Nadam/layer_normalization_1/gamma/v/Read/ReadVariableOp6Nadam/layer_normalization_1/beta/v/Read/ReadVariableOp*Nadam/dense_4/kernel/v/Read/ReadVariableOp(Nadam/dense_4/bias/v/Read/ReadVariableOp*Nadam/dense_5/kernel/v/Read/ReadVariableOp(Nadam/dense_5/bias/v/Read/ReadVariableOp*Nadam/dense_6/kernel/v/Read/ReadVariableOp(Nadam/dense_6/bias/v/Read/ReadVariableOp*Nadam/dense_7/kernel/v/Read/ReadVariableOp(Nadam/dense_7/bias/v/Read/ReadVariableOp3Nadam/gru_1/gru_cell_1/kernel/v/Read/ReadVariableOp=Nadam/gru_1/gru_cell_1/recurrent_kernel/v/Read/ReadVariableOp1Nadam/gru_1/gru_cell_1/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*5J 8? *(
f#R!
__inference__traced_save_225990
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization_1/gammalayer_normalization_1/betadense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachegru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kernelgru_1/gru_cell_1/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negatives#Nadam/layer_normalization_1/gamma/m"Nadam/layer_normalization_1/beta/mNadam/dense_4/kernel/mNadam/dense_4/bias/mNadam/dense_5/kernel/mNadam/dense_5/bias/mNadam/dense_6/kernel/mNadam/dense_6/bias/mNadam/dense_7/kernel/mNadam/dense_7/bias/mNadam/gru_1/gru_cell_1/kernel/m)Nadam/gru_1/gru_cell_1/recurrent_kernel/mNadam/gru_1/gru_cell_1/bias/m#Nadam/layer_normalization_1/gamma/v"Nadam/layer_normalization_1/beta/vNadam/dense_4/kernel/vNadam/dense_4/bias/vNadam/dense_5/kernel/vNadam/dense_5/bias/vNadam/dense_6/kernel/vNadam/dense_6/bias/vNadam/dense_7/kernel/vNadam/dense_7/bias/vNadam/gru_1/gru_cell_1/kernel/v)Nadam/gru_1/gru_cell_1/recurrent_kernel/vNadam/gru_1/gru_cell_1/bias/v*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*5J 8? *+
f&R$
"__inference__traced_restore_226153??*
?

?
(__inference_model_1_layer_call_fn_224169
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2232682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?	
?
gru_1_while_cond_223558(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_223558___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_223558___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_223558___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_223558___redundant_placeholder3
gru_1_while_identity
?
gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
gru_1/while/Lesso
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_1/while/Identity"5
gru_1_while_identitygru_1/while/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
C__inference_dense_6_layer_call_and_return_conditional_losses_223144

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Selu?
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_4_layer_call_and_return_conditional_losses_225487

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Selu?
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_222189
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_222189___redundant_placeholder04
0while_while_cond_222189___redundant_placeholder14
0while_while_cond_222189___redundant_placeholder24
0while_while_cond_222189___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
while_body_225257
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_00
,while_gru_cell_1_readvariableop_1_resource_00
,while_gru_cell_1_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource.
*while_gru_cell_1_readvariableop_1_resource.
*while_gru_cell_1_readvariableop_4_resource??while/gru_cell_1/ReadVariableOp?!while/gru_cell_1/ReadVariableOp_1?!while/gru_cell_1/ReadVariableOp_2?!while/gru_cell_1/ReadVariableOp_3?!while/gru_cell_1/ReadVariableOp_4?!while/gru_cell_1/ReadVariableOp_5?!while/gru_cell_1/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_1/ones_like/Shape?
 while/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_1/ones_like/Const?
while/gru_cell_1/ones_likeFill)while/gru_cell_1/ones_like/Shape:output:0)while/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/ones_like?
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_1/ReadVariableOp?
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_1/unstack?
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_1?
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_1/strided_slice/stack?
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice/stack_1?
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_1/strided_slice/stack_2?
while/gru_cell_1/strided_sliceStridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_1/strided_slice?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul?
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_2?
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_1/stack?
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_1/stack_1?
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_1/stack_2?
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_1?
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_1?
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_3?
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_1/strided_slice_2/stack?
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_1/strided_slice_2/stack_1?
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_2/stack_2?
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_2?
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_2?
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_3/stack?
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_3/stack_1?
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_3/stack_2?
 while/gru_cell_1/strided_slice_3StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_3?
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd?
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_4/stack?
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_4/stack_1?
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_4/stack_2?
 while/gru_cell_1/strided_slice_4StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_4?
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_1?
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_5/stack?
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_1/strided_slice_5/stack_1?
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_5/stack_2?
 while/gru_cell_1/strided_slice_5StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_1/strided_slice_5?
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_2?
while/gru_cell_1/mulMulwhile_placeholder_2#while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul?
while/gru_cell_1/mul_1Mulwhile_placeholder_2#while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_1?
while/gru_cell_1/mul_2Mulwhile_placeholder_2#while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_2?
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_4?
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_1/strided_slice_6/stack?
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_1/strided_slice_6/stack_1?
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_6/stack_2?
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_6?
while/gru_cell_1/MatMul_3MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_3?
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_5?
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_7/stack?
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_7/stack_1?
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_7/stack_2?
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_7?
while/gru_cell_1/MatMul_4MatMulwhile/gru_cell_1/mul_1:z:0)while/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_4?
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_8/stack?
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_8/stack_1?
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_8/stack_2?
 while/gru_cell_1/strided_slice_8StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_8?
while/gru_cell_1/BiasAdd_3BiasAdd#while/gru_cell_1/MatMul_3:product:0)while/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_3?
&while/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_9/stack?
(while/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_9/stack_1?
(while/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_9/stack_2?
 while/gru_cell_1/strided_slice_9StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_9/stack:output:01while/gru_cell_1/strided_slice_9/stack_1:output:01while/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_9?
while/gru_cell_1/BiasAdd_4BiasAdd#while/gru_cell_1/MatMul_4:product:0)while/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_4?
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add?
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid?
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_1?
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid_1?
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_6?
'while/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_1/strided_slice_10/stack?
)while/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_1/strided_slice_10/stack_1?
)while/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_1/strided_slice_10/stack_2?
!while/gru_cell_1/strided_slice_10StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:00while/gru_cell_1/strided_slice_10/stack:output:02while/gru_cell_1/strided_slice_10/stack_1:output:02while/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_1/strided_slice_10?
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul_2:z:0*while/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_5?
'while/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_1/strided_slice_11/stack?
)while/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_1/strided_slice_11/stack_1?
)while/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_1/strided_slice_11/stack_2?
!while/gru_cell_1/strided_slice_11StridedSlice!while/gru_cell_1/unstack:output:10while/gru_cell_1/strided_slice_11/stack:output:02while/gru_cell_1/strided_slice_11/stack_1:output:02while/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_1/strided_slice_11?
while/gru_cell_1/BiasAdd_5BiasAdd#while/gru_cell_1/MatMul_5:product:0*while/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_5?
while/gru_cell_1/mul_3Mulwhile/gru_cell_1/Sigmoid_1:y:0#while/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_3?
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0while/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_2?
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Tanh?
while/gru_cell_1/mul_4Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_4u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_1/sub/x?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/sub?
while/gru_cell_1/mul_5Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_5?
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_4:z:0while/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_1_readvariableop_1_resource,while_gru_cell_1_readvariableop_1_resource_0"Z
*while_gru_cell_1_readvariableop_4_resource,while_gru_cell_1_readvariableop_4_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
}
(__inference_dense_5_layer_call_fn_225529

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2231172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_222307
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_222307___redundant_placeholder04
0while_while_cond_222307___redundant_placeholder14
0while_while_cond_222307___redundant_placeholder24
0while_while_cond_222307___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
s
I__inference_concatenate_1_layer_call_and_return_conditional_losses_223097

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_221835

inputs

states
readvariableop_resource
readvariableop_1_resource
readvariableop_4_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6X
ones_like/ShapeShapestates*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const?
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_2/Mul_1y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slicem
MatMulMatMulinputsstrided_slice:output:0*
T0*(
_output_shapes
:??????????2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slice_1s
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slice_2s
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack}
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
strided_slice_3|
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddy
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_4/stack}
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_1y
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_2_
mulMulstatesdropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mule
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_1e
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_2?
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_6t
MatMul_3MatMulmul:z:0strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2

MatMul_3?
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_7v
MatMul_4MatMul	mul_1:z:0strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack}
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
strided_slice_8?
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_3y
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_9/stack}
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
strided_slice_9?
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_4l
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidr
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1?
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_6?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_10w
MatMul_5MatMul	mul_2:z:0strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5{
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
strided_slice_11?
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_5k
mul_3MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
mul_3i
add_2AddV2BiasAdd_2:output:0	mul_3:z:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh]
mul_4MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
mul_4S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sub[
mul_5Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_5`
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:??????????2
add_3?
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????	:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_6:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?	
?
C__inference_dense_7_layer_call_and_return_conditional_losses_225560

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_224349
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_224349___redundant_placeholder04
0while_while_cond_224349___redundant_placeholder14
0while_while_cond_224349___redundant_placeholder24
0while_while_cond_224349___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
޴
?
A__inference_gru_1_layer_call_and_return_conditional_losses_225403

inputs&
"gru_cell_1_readvariableop_resource(
$gru_cell_1_readvariableop_1_resource(
$gru_cell_1_readvariableop_4_resource
identity??gru_cell_1/ReadVariableOp?gru_cell_1/ReadVariableOp_1?gru_cell_1/ReadVariableOp_2?gru_cell_1/ReadVariableOp_3?gru_cell_1/ReadVariableOp_4?gru_cell_1/ReadVariableOp_5?gru_cell_1/ReadVariableOp_6?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????	2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
strided_slice_2v
gru_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_1/ones_like/Shape}
gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/ones_like/Const?
gru_cell_1/ones_likeFill#gru_cell_1/ones_like/Shape:output:0#gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/ones_like?
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_1/ReadVariableOp?
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_1/unstack?
gru_cell_1/ReadVariableOp_1ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_1?
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_1/strided_slice/stack?
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice/stack_1?
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_1/strided_slice/stack_2?
gru_cell_1/strided_sliceStridedSlice#gru_cell_1/ReadVariableOp_1:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0!gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul?
gru_cell_1/ReadVariableOp_2ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_2?
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_1/stack?
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_1/stack_1?
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_1/stack_2?
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_1?
gru_cell_1/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_1?
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_3?
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_1/strided_slice_2/stack?
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_1/strided_slice_2/stack_1?
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_2/stack_2?
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_2?
gru_cell_1/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_2?
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_3/stack?
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_3/stack_1?
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_3/stack_2?
gru_cell_1/strided_slice_3StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_3?
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd?
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_4/stack?
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_4/stack_1?
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_4/stack_2?
gru_cell_1/strided_slice_4StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_4?
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_1?
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_5/stack?
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_1/strided_slice_5/stack_1?
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_5/stack_2?
gru_cell_1/strided_slice_5StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_5?
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_2?
gru_cell_1/mulMulzeros:output:0gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul?
gru_cell_1/mul_1Mulzeros:output:0gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_1?
gru_cell_1/mul_2Mulzeros:output:0gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_2?
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_4?
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_1/strided_slice_6/stack?
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_1/strided_slice_6/stack_1?
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_6/stack_2?
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_6?
gru_cell_1/MatMul_3MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_3?
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_5?
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_7/stack?
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_7/stack_1?
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_7/stack_2?
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_7?
gru_cell_1/MatMul_4MatMulgru_cell_1/mul_1:z:0#gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_4?
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_8/stack?
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_8/stack_1?
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_8/stack_2?
gru_cell_1/strided_slice_8StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_8?
gru_cell_1/BiasAdd_3BiasAddgru_cell_1/MatMul_3:product:0#gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_3?
 gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_9/stack?
"gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_9/stack_1?
"gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_9/stack_2?
gru_cell_1/strided_slice_9StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_9/stack:output:0+gru_cell_1/strided_slice_9/stack_1:output:0+gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_9?
gru_cell_1/BiasAdd_4BiasAddgru_cell_1/MatMul_4:product:0#gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_4?
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/addz
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid?
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_1?
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid_1?
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_6?
!gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_1/strided_slice_10/stack?
#gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_1/strided_slice_10/stack_1?
#gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_1/strided_slice_10/stack_2?
gru_cell_1/strided_slice_10StridedSlice#gru_cell_1/ReadVariableOp_6:value:0*gru_cell_1/strided_slice_10/stack:output:0,gru_cell_1/strided_slice_10/stack_1:output:0,gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_10?
gru_cell_1/MatMul_5MatMulgru_cell_1/mul_2:z:0$gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_5?
!gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_1/strided_slice_11/stack?
#gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_1/strided_slice_11/stack_1?
#gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_1/strided_slice_11/stack_2?
gru_cell_1/strided_slice_11StridedSlicegru_cell_1/unstack:output:1*gru_cell_1/strided_slice_11/stack:output:0,gru_cell_1/strided_slice_11/stack_1:output:0,gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_11?
gru_cell_1/BiasAdd_5BiasAddgru_cell_1/MatMul_5:product:0$gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_5?
gru_cell_1/mul_3Mulgru_cell_1/Sigmoid_1:y:0gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_3?
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_2s
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Tanh?
gru_cell_1/mul_4Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_4i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/sub/x?
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/sub?
gru_cell_1/mul_5Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_5?
gru_cell_1/add_3AddV2gru_cell_1/mul_4:z:0gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_1_resource$gru_cell_1_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_225257*
condR
while_cond_225256*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?

C__inference_model_1_layer_call_and_return_conditional_losses_223797
inputs_0
inputs_1,
(gru_1_gru_cell_1_readvariableop_resource.
*gru_1_gru_cell_1_readvariableop_1_resource.
*gru_1_gru_cell_1_readvariableop_4_resource7
3layer_normalization_1_mul_2_readvariableop_resource5
1layer_normalization_1_add_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?gru_1/gru_cell_1/ReadVariableOp?!gru_1/gru_cell_1/ReadVariableOp_1?!gru_1/gru_cell_1/ReadVariableOp_2?!gru_1/gru_cell_1/ReadVariableOp_3?!gru_1/gru_cell_1/ReadVariableOp_4?!gru_1/gru_cell_1/ReadVariableOp_5?!gru_1/gru_cell_1/ReadVariableOp_6?gru_1/while?(layer_normalization_1/add/ReadVariableOp?*layer_normalization_1/mul_2/ReadVariableOpR
gru_1/ShapeShapeinputs_0*
T0*
_output_shapes
:2
gru_1/Shape?
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice/stack?
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_1?
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_2?
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_1/strided_slicei
gru_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_1/zeros/mul/y?
gru_1/zeros/mulMulgru_1/strided_slice:output:0gru_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/mulk
gru_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_1/zeros/Less/y
gru_1/zeros/LessLessgru_1/zeros/mul:z:0gru_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/Lesso
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_1/zeros/packed/1?
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_1/zeros/packedk
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_1/zeros/Const?
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_1/zeros?
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_1/transpose/perm?
gru_1/transpose	Transposeinputs_0gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????	2
gru_1/transposea
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
:2
gru_1/Shape_1?
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_1/stack?
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_1/stack_1?
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_1/stack_2?
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_1/strided_slice_1?
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru_1/TensorArrayV2/element_shape?
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_1/TensorArrayV2?
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2=
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_1/TensorArrayUnstack/TensorListFromTensor?
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_2/stack?
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_2/stack_1?
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_2/stack_2?
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
gru_1/strided_slice_2?
 gru_1/gru_cell_1/ones_like/ShapeShapegru_1/zeros:output:0*
T0*
_output_shapes
:2"
 gru_1/gru_cell_1/ones_like/Shape?
 gru_1/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 gru_1/gru_cell_1/ones_like/Const?
gru_1/gru_cell_1/ones_likeFill)gru_1/gru_cell_1/ones_like/Shape:output:0)gru_1/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/ones_like?
gru_1/gru_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
gru_1/gru_cell_1/dropout/Const?
gru_1/gru_cell_1/dropout/MulMul#gru_1/gru_cell_1/ones_like:output:0'gru_1/gru_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/dropout/Mul?
gru_1/gru_cell_1/dropout/ShapeShape#gru_1/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2 
gru_1/gru_cell_1/dropout/Shape?
5gru_1/gru_cell_1/dropout/random_uniform/RandomUniformRandomUniform'gru_1/gru_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5gru_1/gru_cell_1/dropout/random_uniform/RandomUniform?
'gru_1/gru_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'gru_1/gru_cell_1/dropout/GreaterEqual/y?
%gru_1/gru_cell_1/dropout/GreaterEqualGreaterEqual>gru_1/gru_cell_1/dropout/random_uniform/RandomUniform:output:00gru_1/gru_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%gru_1/gru_cell_1/dropout/GreaterEqual?
gru_1/gru_cell_1/dropout/CastCast)gru_1/gru_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_1/gru_cell_1/dropout/Cast?
gru_1/gru_cell_1/dropout/Mul_1Mul gru_1/gru_cell_1/dropout/Mul:z:0!gru_1/gru_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2 
gru_1/gru_cell_1/dropout/Mul_1?
 gru_1/gru_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 gru_1/gru_cell_1/dropout_1/Const?
gru_1/gru_cell_1/dropout_1/MulMul#gru_1/gru_cell_1/ones_like:output:0)gru_1/gru_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
gru_1/gru_cell_1/dropout_1/Mul?
 gru_1/gru_cell_1/dropout_1/ShapeShape#gru_1/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 gru_1/gru_cell_1/dropout_1/Shape?
7gru_1/gru_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform)gru_1/gru_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2憧29
7gru_1/gru_cell_1/dropout_1/random_uniform/RandomUniform?
)gru_1/gru_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)gru_1/gru_cell_1/dropout_1/GreaterEqual/y?
'gru_1/gru_cell_1/dropout_1/GreaterEqualGreaterEqual@gru_1/gru_cell_1/dropout_1/random_uniform/RandomUniform:output:02gru_1/gru_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'gru_1/gru_cell_1/dropout_1/GreaterEqual?
gru_1/gru_cell_1/dropout_1/CastCast+gru_1/gru_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
gru_1/gru_cell_1/dropout_1/Cast?
 gru_1/gru_cell_1/dropout_1/Mul_1Mul"gru_1/gru_cell_1/dropout_1/Mul:z:0#gru_1/gru_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 gru_1/gru_cell_1/dropout_1/Mul_1?
 gru_1/gru_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 gru_1/gru_cell_1/dropout_2/Const?
gru_1/gru_cell_1/dropout_2/MulMul#gru_1/gru_cell_1/ones_like:output:0)gru_1/gru_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
gru_1/gru_cell_1/dropout_2/Mul?
 gru_1/gru_cell_1/dropout_2/ShapeShape#gru_1/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 gru_1/gru_cell_1/dropout_2/Shape?
7gru_1/gru_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform)gru_1/gru_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???29
7gru_1/gru_cell_1/dropout_2/random_uniform/RandomUniform?
)gru_1/gru_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)gru_1/gru_cell_1/dropout_2/GreaterEqual/y?
'gru_1/gru_cell_1/dropout_2/GreaterEqualGreaterEqual@gru_1/gru_cell_1/dropout_2/random_uniform/RandomUniform:output:02gru_1/gru_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'gru_1/gru_cell_1/dropout_2/GreaterEqual?
gru_1/gru_cell_1/dropout_2/CastCast+gru_1/gru_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
gru_1/gru_cell_1/dropout_2/Cast?
 gru_1/gru_cell_1/dropout_2/Mul_1Mul"gru_1/gru_cell_1/dropout_2/Mul:z:0#gru_1/gru_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 gru_1/gru_cell_1/dropout_2/Mul_1?
gru_1/gru_cell_1/ReadVariableOpReadVariableOp(gru_1_gru_cell_1_readvariableop_resource*
_output_shapes
:	?*
dtype02!
gru_1/gru_cell_1/ReadVariableOp?
gru_1/gru_cell_1/unstackUnpack'gru_1/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_1/gru_cell_1/unstack?
!gru_1/gru_cell_1/ReadVariableOp_1ReadVariableOp*gru_1_gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_1?
$gru_1/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$gru_1/gru_cell_1/strided_slice/stack?
&gru_1/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_1/gru_cell_1/strided_slice/stack_1?
&gru_1/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru_1/gru_cell_1/strided_slice/stack_2?
gru_1/gru_cell_1/strided_sliceStridedSlice)gru_1/gru_cell_1/ReadVariableOp_1:value:0-gru_1/gru_cell_1/strided_slice/stack:output:0/gru_1/gru_cell_1/strided_slice/stack_1:output:0/gru_1/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
gru_1/gru_cell_1/strided_slice?
gru_1/gru_cell_1/MatMulMatMulgru_1/strided_slice_2:output:0'gru_1/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul?
!gru_1/gru_cell_1/ReadVariableOp_2ReadVariableOp*gru_1_gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_2?
&gru_1/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_1/gru_cell_1/strided_slice_1/stack?
(gru_1/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(gru_1/gru_cell_1/strided_slice_1/stack_1?
(gru_1/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_1/gru_cell_1/strided_slice_1/stack_2?
 gru_1/gru_cell_1/strided_slice_1StridedSlice)gru_1/gru_cell_1/ReadVariableOp_2:value:0/gru_1/gru_cell_1/strided_slice_1/stack:output:01gru_1/gru_cell_1/strided_slice_1/stack_1:output:01gru_1/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 gru_1/gru_cell_1/strided_slice_1?
gru_1/gru_cell_1/MatMul_1MatMulgru_1/strided_slice_2:output:0)gru_1/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul_1?
!gru_1/gru_cell_1/ReadVariableOp_3ReadVariableOp*gru_1_gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_3?
&gru_1/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&gru_1/gru_cell_1/strided_slice_2/stack?
(gru_1/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(gru_1/gru_cell_1/strided_slice_2/stack_1?
(gru_1/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_1/gru_cell_1/strided_slice_2/stack_2?
 gru_1/gru_cell_1/strided_slice_2StridedSlice)gru_1/gru_cell_1/ReadVariableOp_3:value:0/gru_1/gru_cell_1/strided_slice_2/stack:output:01gru_1/gru_cell_1/strided_slice_2/stack_1:output:01gru_1/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 gru_1/gru_cell_1/strided_slice_2?
gru_1/gru_cell_1/MatMul_2MatMulgru_1/strided_slice_2:output:0)gru_1/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul_2?
&gru_1/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&gru_1/gru_cell_1/strided_slice_3/stack?
(gru_1/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_1/gru_cell_1/strided_slice_3/stack_1?
(gru_1/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_1/gru_cell_1/strided_slice_3/stack_2?
 gru_1/gru_cell_1/strided_slice_3StridedSlice!gru_1/gru_cell_1/unstack:output:0/gru_1/gru_cell_1/strided_slice_3/stack:output:01gru_1/gru_cell_1/strided_slice_3/stack_1:output:01gru_1/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 gru_1/gru_cell_1/strided_slice_3?
gru_1/gru_cell_1/BiasAddBiasAdd!gru_1/gru_cell_1/MatMul:product:0)gru_1/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd?
&gru_1/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_1/gru_cell_1/strided_slice_4/stack?
(gru_1/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_1/gru_cell_1/strided_slice_4/stack_1?
(gru_1/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_1/gru_cell_1/strided_slice_4/stack_2?
 gru_1/gru_cell_1/strided_slice_4StridedSlice!gru_1/gru_cell_1/unstack:output:0/gru_1/gru_cell_1/strided_slice_4/stack:output:01gru_1/gru_cell_1/strided_slice_4/stack_1:output:01gru_1/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 gru_1/gru_cell_1/strided_slice_4?
gru_1/gru_cell_1/BiasAdd_1BiasAdd#gru_1/gru_cell_1/MatMul_1:product:0)gru_1/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd_1?
&gru_1/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_1/gru_cell_1/strided_slice_5/stack?
(gru_1/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(gru_1/gru_cell_1/strided_slice_5/stack_1?
(gru_1/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_1/gru_cell_1/strided_slice_5/stack_2?
 gru_1/gru_cell_1/strided_slice_5StridedSlice!gru_1/gru_cell_1/unstack:output:0/gru_1/gru_cell_1/strided_slice_5/stack:output:01gru_1/gru_cell_1/strided_slice_5/stack_1:output:01gru_1/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 gru_1/gru_cell_1/strided_slice_5?
gru_1/gru_cell_1/BiasAdd_2BiasAdd#gru_1/gru_cell_1/MatMul_2:product:0)gru_1/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd_2?
gru_1/gru_cell_1/mulMulgru_1/zeros:output:0"gru_1/gru_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul?
gru_1/gru_cell_1/mul_1Mulgru_1/zeros:output:0$gru_1/gru_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul_1?
gru_1/gru_cell_1/mul_2Mulgru_1/zeros:output:0$gru_1/gru_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul_2?
!gru_1/gru_cell_1/ReadVariableOp_4ReadVariableOp*gru_1_gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_4?
&gru_1/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru_1/gru_cell_1/strided_slice_6/stack?
(gru_1/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(gru_1/gru_cell_1/strided_slice_6/stack_1?
(gru_1/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_1/gru_cell_1/strided_slice_6/stack_2?
 gru_1/gru_cell_1/strided_slice_6StridedSlice)gru_1/gru_cell_1/ReadVariableOp_4:value:0/gru_1/gru_cell_1/strided_slice_6/stack:output:01gru_1/gru_cell_1/strided_slice_6/stack_1:output:01gru_1/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 gru_1/gru_cell_1/strided_slice_6?
gru_1/gru_cell_1/MatMul_3MatMulgru_1/gru_cell_1/mul:z:0)gru_1/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul_3?
!gru_1/gru_cell_1/ReadVariableOp_5ReadVariableOp*gru_1_gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_5?
&gru_1/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_1/gru_cell_1/strided_slice_7/stack?
(gru_1/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(gru_1/gru_cell_1/strided_slice_7/stack_1?
(gru_1/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_1/gru_cell_1/strided_slice_7/stack_2?
 gru_1/gru_cell_1/strided_slice_7StridedSlice)gru_1/gru_cell_1/ReadVariableOp_5:value:0/gru_1/gru_cell_1/strided_slice_7/stack:output:01gru_1/gru_cell_1/strided_slice_7/stack_1:output:01gru_1/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 gru_1/gru_cell_1/strided_slice_7?
gru_1/gru_cell_1/MatMul_4MatMulgru_1/gru_cell_1/mul_1:z:0)gru_1/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul_4?
&gru_1/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&gru_1/gru_cell_1/strided_slice_8/stack?
(gru_1/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_1/gru_cell_1/strided_slice_8/stack_1?
(gru_1/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_1/gru_cell_1/strided_slice_8/stack_2?
 gru_1/gru_cell_1/strided_slice_8StridedSlice!gru_1/gru_cell_1/unstack:output:1/gru_1/gru_cell_1/strided_slice_8/stack:output:01gru_1/gru_cell_1/strided_slice_8/stack_1:output:01gru_1/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 gru_1/gru_cell_1/strided_slice_8?
gru_1/gru_cell_1/BiasAdd_3BiasAdd#gru_1/gru_cell_1/MatMul_3:product:0)gru_1/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd_3?
&gru_1/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_1/gru_cell_1/strided_slice_9/stack?
(gru_1/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_1/gru_cell_1/strided_slice_9/stack_1?
(gru_1/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_1/gru_cell_1/strided_slice_9/stack_2?
 gru_1/gru_cell_1/strided_slice_9StridedSlice!gru_1/gru_cell_1/unstack:output:1/gru_1/gru_cell_1/strided_slice_9/stack:output:01gru_1/gru_cell_1/strided_slice_9/stack_1:output:01gru_1/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 gru_1/gru_cell_1/strided_slice_9?
gru_1/gru_cell_1/BiasAdd_4BiasAdd#gru_1/gru_cell_1/MatMul_4:product:0)gru_1/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd_4?
gru_1/gru_cell_1/addAddV2!gru_1/gru_cell_1/BiasAdd:output:0#gru_1/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/add?
gru_1/gru_cell_1/SigmoidSigmoidgru_1/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/Sigmoid?
gru_1/gru_cell_1/add_1AddV2#gru_1/gru_cell_1/BiasAdd_1:output:0#gru_1/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/add_1?
gru_1/gru_cell_1/Sigmoid_1Sigmoidgru_1/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/Sigmoid_1?
!gru_1/gru_cell_1/ReadVariableOp_6ReadVariableOp*gru_1_gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_6?
'gru_1/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'gru_1/gru_cell_1/strided_slice_10/stack?
)gru_1/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)gru_1/gru_cell_1/strided_slice_10/stack_1?
)gru_1/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)gru_1/gru_cell_1/strided_slice_10/stack_2?
!gru_1/gru_cell_1/strided_slice_10StridedSlice)gru_1/gru_cell_1/ReadVariableOp_6:value:00gru_1/gru_cell_1/strided_slice_10/stack:output:02gru_1/gru_cell_1/strided_slice_10/stack_1:output:02gru_1/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!gru_1/gru_cell_1/strided_slice_10?
gru_1/gru_cell_1/MatMul_5MatMulgru_1/gru_cell_1/mul_2:z:0*gru_1/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul_5?
'gru_1/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'gru_1/gru_cell_1/strided_slice_11/stack?
)gru_1/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)gru_1/gru_cell_1/strided_slice_11/stack_1?
)gru_1/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)gru_1/gru_cell_1/strided_slice_11/stack_2?
!gru_1/gru_cell_1/strided_slice_11StridedSlice!gru_1/gru_cell_1/unstack:output:10gru_1/gru_cell_1/strided_slice_11/stack:output:02gru_1/gru_cell_1/strided_slice_11/stack_1:output:02gru_1/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!gru_1/gru_cell_1/strided_slice_11?
gru_1/gru_cell_1/BiasAdd_5BiasAdd#gru_1/gru_cell_1/MatMul_5:product:0*gru_1/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd_5?
gru_1/gru_cell_1/mul_3Mulgru_1/gru_cell_1/Sigmoid_1:y:0#gru_1/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul_3?
gru_1/gru_cell_1/add_2AddV2#gru_1/gru_cell_1/BiasAdd_2:output:0gru_1/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/add_2?
gru_1/gru_cell_1/TanhTanhgru_1/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/Tanh?
gru_1/gru_cell_1/mul_4Mulgru_1/gru_cell_1/Sigmoid:y:0gru_1/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul_4u
gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_1/gru_cell_1/sub/x?
gru_1/gru_cell_1/subSubgru_1/gru_cell_1/sub/x:output:0gru_1/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/sub?
gru_1/gru_cell_1/mul_5Mulgru_1/gru_cell_1/sub:z:0gru_1/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul_5?
gru_1/gru_cell_1/add_3AddV2gru_1/gru_cell_1/mul_4:z:0gru_1/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/add_3?
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2%
#gru_1/TensorArrayV2_1/element_shape?
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_1/TensorArrayV2_1Z

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_1/time?
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru_1/while/maximum_iterationsv
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_1/while/loop_counter?
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_1_readvariableop_resource*gru_1_gru_cell_1_readvariableop_1_resource*gru_1_gru_cell_1_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*#
bodyR
gru_1_while_body_223559*#
condR
gru_1_while_cond_223558*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_1/while?
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   28
6gru_1/TensorArrayV2Stack/TensorListStack/element_shape?
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02*
(gru_1/TensorArrayV2Stack/TensorListStack?
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_1/strided_slice_3/stack?
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_3/stack_1?
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_3/stack_2?
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_1/strided_slice_3?
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_1/transpose_1/perm?
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_1/transpose_1r
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_1/runtime?
layer_normalization_1/ShapeShapegru_1/strided_slice_3:output:0*
T0*
_output_shapes
:2
layer_normalization_1/Shape?
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_1/strided_slice/stack?
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_1?
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_2?
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_1/strided_slice|
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul/x?
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul?
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice_1/stack?
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_1?
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_2?
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_1/strided_slice_1?
layer_normalization_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul_1/x?
layer_normalization_1/mul_1Mul&layer_normalization_1/mul_1/x:output:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul_1?
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/0?
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/3?
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul:z:0layer_normalization_1/mul_1:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_1/Reshape/shape?
layer_normalization_1/ReshapeReshapegru_1/strided_slice_3:output:0,layer_normalization_1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_1/Reshape
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_1/Const?
layer_normalization_1/Fill/dimsPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_1/Fill/dims?
layer_normalization_1/FillFill(layer_normalization_1/Fill/dims:output:0$layer_normalization_1/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_1/Fill?
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_1/Const_1?
!layer_normalization_1/Fill_1/dimsPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_1/Fill_1/dims?
layer_normalization_1/Fill_1Fill*layer_normalization_1/Fill_1/dims:output:0&layer_normalization_1/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_1/Fill_1?
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_2?
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_3?
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/Fill:output:0%layer_normalization_1/Fill_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_1/FusedBatchNormV3?
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*(
_output_shapes
:??????????2!
layer_normalization_1/Reshape_1?
*layer_normalization_1/mul_2/ReadVariableOpReadVariableOp3layer_normalization_1_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*layer_normalization_1/mul_2/ReadVariableOp?
layer_normalization_1/mul_2Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_1/mul_2?
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization_1/add/ReadVariableOp?
layer_normalization_1/addAddV2layer_normalization_1/mul_2:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_1/add?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulinputs_1%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddq
dense_4/SeluSeludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_4/Selux
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2layer_normalization_1/add:z:0dense_4/Selu:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_1/concat?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulconcatenate_1/concat:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/BiasAddq
dense_5/SeluSeludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_5/Selu?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldense_5/Selu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddq
dense_6/SeluSeludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Selu?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Selu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/BiasAddy
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_7/Sigmoid?
IdentityIdentitydense_7/Sigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^gru_1/gru_cell_1/ReadVariableOp"^gru_1/gru_cell_1/ReadVariableOp_1"^gru_1/gru_cell_1/ReadVariableOp_2"^gru_1/gru_cell_1/ReadVariableOp_3"^gru_1/gru_cell_1/ReadVariableOp_4"^gru_1/gru_cell_1/ReadVariableOp_5"^gru_1/gru_cell_1/ReadVariableOp_6^gru_1/while)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2B
gru_1/gru_cell_1/ReadVariableOpgru_1/gru_cell_1/ReadVariableOp2F
!gru_1/gru_cell_1/ReadVariableOp_1!gru_1/gru_cell_1/ReadVariableOp_12F
!gru_1/gru_cell_1/ReadVariableOp_2!gru_1/gru_cell_1/ReadVariableOp_22F
!gru_1/gru_cell_1/ReadVariableOp_3!gru_1/gru_cell_1/ReadVariableOp_32F
!gru_1/gru_cell_1/ReadVariableOp_4!gru_1/gru_cell_1/ReadVariableOp_42F
!gru_1/gru_cell_1/ReadVariableOp_5!gru_1/gru_cell_1/ReadVariableOp_52F
!gru_1/gru_cell_1/ReadVariableOp_6!gru_1/gru_cell_1/ReadVariableOp_62
gru_1/whilegru_1/while2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_2/ReadVariableOp*layer_normalization_1/mul_2/ReadVariableOp:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
while_cond_225256
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_225256___redundant_placeholder04
0while_while_cond_225256___redundant_placeholder14
0while_while_cond_225256___redundant_placeholder24
0while_while_cond_225256___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
A__inference_gru_1_layer_call_and_return_conditional_losses_224791
inputs_0&
"gru_cell_1_readvariableop_resource(
$gru_cell_1_readvariableop_1_resource(
$gru_cell_1_readvariableop_4_resource
identity??gru_cell_1/ReadVariableOp?gru_cell_1/ReadVariableOp_1?gru_cell_1/ReadVariableOp_2?gru_cell_1/ReadVariableOp_3?gru_cell_1/ReadVariableOp_4?gru_cell_1/ReadVariableOp_5?gru_cell_1/ReadVariableOp_6?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????	2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
strided_slice_2v
gru_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_1/ones_like/Shape}
gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/ones_like/Const?
gru_cell_1/ones_likeFill#gru_cell_1/ones_like/Shape:output:0#gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/ones_like?
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_1/ReadVariableOp?
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_1/unstack?
gru_cell_1/ReadVariableOp_1ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_1?
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_1/strided_slice/stack?
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice/stack_1?
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_1/strided_slice/stack_2?
gru_cell_1/strided_sliceStridedSlice#gru_cell_1/ReadVariableOp_1:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0!gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul?
gru_cell_1/ReadVariableOp_2ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_2?
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_1/stack?
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_1/stack_1?
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_1/stack_2?
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_1?
gru_cell_1/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_1?
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_3?
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_1/strided_slice_2/stack?
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_1/strided_slice_2/stack_1?
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_2/stack_2?
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_2?
gru_cell_1/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_2?
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_3/stack?
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_3/stack_1?
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_3/stack_2?
gru_cell_1/strided_slice_3StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_3?
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd?
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_4/stack?
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_4/stack_1?
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_4/stack_2?
gru_cell_1/strided_slice_4StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_4?
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_1?
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_5/stack?
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_1/strided_slice_5/stack_1?
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_5/stack_2?
gru_cell_1/strided_slice_5StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_5?
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_2?
gru_cell_1/mulMulzeros:output:0gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul?
gru_cell_1/mul_1Mulzeros:output:0gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_1?
gru_cell_1/mul_2Mulzeros:output:0gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_2?
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_4?
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_1/strided_slice_6/stack?
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_1/strided_slice_6/stack_1?
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_6/stack_2?
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_6?
gru_cell_1/MatMul_3MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_3?
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_5?
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_7/stack?
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_7/stack_1?
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_7/stack_2?
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_7?
gru_cell_1/MatMul_4MatMulgru_cell_1/mul_1:z:0#gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_4?
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_8/stack?
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_8/stack_1?
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_8/stack_2?
gru_cell_1/strided_slice_8StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_8?
gru_cell_1/BiasAdd_3BiasAddgru_cell_1/MatMul_3:product:0#gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_3?
 gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_9/stack?
"gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_9/stack_1?
"gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_9/stack_2?
gru_cell_1/strided_slice_9StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_9/stack:output:0+gru_cell_1/strided_slice_9/stack_1:output:0+gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_9?
gru_cell_1/BiasAdd_4BiasAddgru_cell_1/MatMul_4:product:0#gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_4?
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/addz
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid?
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_1?
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid_1?
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_6?
!gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_1/strided_slice_10/stack?
#gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_1/strided_slice_10/stack_1?
#gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_1/strided_slice_10/stack_2?
gru_cell_1/strided_slice_10StridedSlice#gru_cell_1/ReadVariableOp_6:value:0*gru_cell_1/strided_slice_10/stack:output:0,gru_cell_1/strided_slice_10/stack_1:output:0,gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_10?
gru_cell_1/MatMul_5MatMulgru_cell_1/mul_2:z:0$gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_5?
!gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_1/strided_slice_11/stack?
#gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_1/strided_slice_11/stack_1?
#gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_1/strided_slice_11/stack_2?
gru_cell_1/strided_slice_11StridedSlicegru_cell_1/unstack:output:1*gru_cell_1/strided_slice_11/stack:output:0,gru_cell_1/strided_slice_11/stack_1:output:0,gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_11?
gru_cell_1/BiasAdd_5BiasAddgru_cell_1/MatMul_5:product:0$gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_5?
gru_cell_1/mul_3Mulgru_cell_1/Sigmoid_1:y:0gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_3?
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_2s
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Tanh?
gru_cell_1/mul_4Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_4i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/sub/x?
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/sub?
gru_cell_1/mul_5Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_5?
gru_cell_1/add_3AddV2gru_cell_1/mul_4:z:0gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_1_resource$gru_cell_1_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_224645*
condR
while_cond_224644*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62
whilewhile:^ Z
4
_output_shapes"
 :??????????????????	
"
_user_specified_name
inputs/0
?
?
&__inference_gru_1_layer_call_fn_225414

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_2227042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?"
?
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_223047

inputs!
mul_2_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3z
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*(
_output_shapes
:??????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
mul_2/ReadVariableOpz
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpm
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^mul_2/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
while_body_222308
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_1_222330_0
while_gru_cell_1_222332_0
while_gru_cell_1_222334_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_1_222330
while_gru_cell_1_222332
while_gru_cell_1_222334??(while/gru_cell_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_222330_0while_gru_cell_1_222332_0while_gru_cell_1_222334_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_2219312*
(while/gru_cell_1/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1)^while/gru_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4"4
while_gru_cell_1_222330while_gru_cell_1_222330_0"4
while_gru_cell_1_222332while_gru_cell_1_222332_0"4
while_gru_cell_1_222334while_gru_cell_1_222334_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2T
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?"
?
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_225467

inputs!
mul_2_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3z
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*(
_output_shapes
:??????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
mul_2/ReadVariableOpz
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpm
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^mul_2/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
A__inference_gru_1_layer_call_and_return_conditional_losses_225132

inputs&
"gru_cell_1_readvariableop_resource(
$gru_cell_1_readvariableop_1_resource(
$gru_cell_1_readvariableop_4_resource
identity??gru_cell_1/ReadVariableOp?gru_cell_1/ReadVariableOp_1?gru_cell_1/ReadVariableOp_2?gru_cell_1/ReadVariableOp_3?gru_cell_1/ReadVariableOp_4?gru_cell_1/ReadVariableOp_5?gru_cell_1/ReadVariableOp_6?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????	2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
strided_slice_2v
gru_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_1/ones_like/Shape}
gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/ones_like/Const?
gru_cell_1/ones_likeFill#gru_cell_1/ones_like/Shape:output:0#gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/ones_likey
gru_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_1/dropout/Const?
gru_cell_1/dropout/MulMulgru_cell_1/ones_like:output:0!gru_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout/Mul?
gru_cell_1/dropout/ShapeShapegru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_1/dropout/Shape?
/gru_cell_1/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??|21
/gru_cell_1/dropout/random_uniform/RandomUniform?
!gru_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell_1/dropout/GreaterEqual/y?
gru_cell_1/dropout/GreaterEqualGreaterEqual8gru_cell_1/dropout/random_uniform/RandomUniform:output:0*gru_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell_1/dropout/GreaterEqual?
gru_cell_1/dropout/CastCast#gru_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_1/dropout/Cast?
gru_cell_1/dropout/Mul_1Mulgru_cell_1/dropout/Mul:z:0gru_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout/Mul_1}
gru_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_1/dropout_1/Const?
gru_cell_1/dropout_1/MulMulgru_cell_1/ones_like:output:0#gru_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_1/Mul?
gru_cell_1/dropout_1/ShapeShapegru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_1/dropout_1/Shape?
1gru_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???23
1gru_cell_1/dropout_1/random_uniform/RandomUniform?
#gru_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_1/dropout_1/GreaterEqual/y?
!gru_cell_1/dropout_1/GreaterEqualGreaterEqual:gru_cell_1/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_1/dropout_1/GreaterEqual?
gru_cell_1/dropout_1/CastCast%gru_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_1/dropout_1/Cast?
gru_cell_1/dropout_1/Mul_1Mulgru_cell_1/dropout_1/Mul:z:0gru_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_1/Mul_1}
gru_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_1/dropout_2/Const?
gru_cell_1/dropout_2/MulMulgru_cell_1/ones_like:output:0#gru_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_2/Mul?
gru_cell_1/dropout_2/ShapeShapegru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_1/dropout_2/Shape?
1gru_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??_23
1gru_cell_1/dropout_2/random_uniform/RandomUniform?
#gru_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_1/dropout_2/GreaterEqual/y?
!gru_cell_1/dropout_2/GreaterEqualGreaterEqual:gru_cell_1/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_1/dropout_2/GreaterEqual?
gru_cell_1/dropout_2/CastCast%gru_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_1/dropout_2/Cast?
gru_cell_1/dropout_2/Mul_1Mulgru_cell_1/dropout_2/Mul:z:0gru_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_2/Mul_1?
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_1/ReadVariableOp?
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_1/unstack?
gru_cell_1/ReadVariableOp_1ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_1?
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_1/strided_slice/stack?
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice/stack_1?
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_1/strided_slice/stack_2?
gru_cell_1/strided_sliceStridedSlice#gru_cell_1/ReadVariableOp_1:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0!gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul?
gru_cell_1/ReadVariableOp_2ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_2?
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_1/stack?
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_1/stack_1?
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_1/stack_2?
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_1?
gru_cell_1/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_1?
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_3?
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_1/strided_slice_2/stack?
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_1/strided_slice_2/stack_1?
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_2/stack_2?
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_2?
gru_cell_1/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_2?
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_3/stack?
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_3/stack_1?
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_3/stack_2?
gru_cell_1/strided_slice_3StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_3?
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd?
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_4/stack?
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_4/stack_1?
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_4/stack_2?
gru_cell_1/strided_slice_4StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_4?
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_1?
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_5/stack?
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_1/strided_slice_5/stack_1?
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_5/stack_2?
gru_cell_1/strided_slice_5StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_5?
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_2?
gru_cell_1/mulMulzeros:output:0gru_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul?
gru_cell_1/mul_1Mulzeros:output:0gru_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_1?
gru_cell_1/mul_2Mulzeros:output:0gru_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_2?
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_4?
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_1/strided_slice_6/stack?
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_1/strided_slice_6/stack_1?
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_6/stack_2?
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_6?
gru_cell_1/MatMul_3MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_3?
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_5?
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_7/stack?
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_7/stack_1?
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_7/stack_2?
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_7?
gru_cell_1/MatMul_4MatMulgru_cell_1/mul_1:z:0#gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_4?
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_8/stack?
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_8/stack_1?
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_8/stack_2?
gru_cell_1/strided_slice_8StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_8?
gru_cell_1/BiasAdd_3BiasAddgru_cell_1/MatMul_3:product:0#gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_3?
 gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_9/stack?
"gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_9/stack_1?
"gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_9/stack_2?
gru_cell_1/strided_slice_9StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_9/stack:output:0+gru_cell_1/strided_slice_9/stack_1:output:0+gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_9?
gru_cell_1/BiasAdd_4BiasAddgru_cell_1/MatMul_4:product:0#gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_4?
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/addz
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid?
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_1?
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid_1?
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_6?
!gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_1/strided_slice_10/stack?
#gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_1/strided_slice_10/stack_1?
#gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_1/strided_slice_10/stack_2?
gru_cell_1/strided_slice_10StridedSlice#gru_cell_1/ReadVariableOp_6:value:0*gru_cell_1/strided_slice_10/stack:output:0,gru_cell_1/strided_slice_10/stack_1:output:0,gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_10?
gru_cell_1/MatMul_5MatMulgru_cell_1/mul_2:z:0$gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_5?
!gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_1/strided_slice_11/stack?
#gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_1/strided_slice_11/stack_1?
#gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_1/strided_slice_11/stack_2?
gru_cell_1/strided_slice_11StridedSlicegru_cell_1/unstack:output:1*gru_cell_1/strided_slice_11/stack:output:0,gru_cell_1/strided_slice_11/stack_1:output:0,gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_11?
gru_cell_1/BiasAdd_5BiasAddgru_cell_1/MatMul_5:product:0$gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_5?
gru_cell_1/mul_3Mulgru_cell_1/Sigmoid_1:y:0gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_3?
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_2s
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Tanh?
gru_cell_1/mul_4Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_4i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/sub/x?
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/sub?
gru_cell_1/mul_5Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_5?
gru_cell_1/add_3AddV2gru_cell_1/mul_4:z:0gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_1_resource$gru_cell_1_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_224962*
condR
while_cond_224961*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?%
?
C__inference_model_1_layer_call_and_return_conditional_losses_223338

inputs
inputs_1
gru_1_223304
gru_1_223306
gru_1_223308 
layer_normalization_1_223311 
layer_normalization_1_223313
dense_4_223316
dense_4_223318
dense_5_223322
dense_5_223324
dense_6_223327
dense_6_223329
dense_7_223332
dense_7_223334
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_223304gru_1_223306gru_1_223308*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_2229752
gru_1/StatefulPartitionedCall?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0layer_normalization_1_223311layer_normalization_1_223313*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_2230472/
-layer_normalization_1/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_4_223316dense_4_223318*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2230742!
dense_4/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*5J 8? *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_2230972
concatenate_1/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_5_223322dense_5_223324*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2231172!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_223327dense_6_223329*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2231442!
dense_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_223332dense_7_223334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2231712!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^gru_1/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
A__inference_gru_1_layer_call_and_return_conditional_losses_222254

inputs
gru_cell_1_222178
gru_cell_1_222180
gru_cell_1_222182
identity??"gru_cell_1/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????	2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
strided_slice_2?
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_1_222178gru_cell_1_222180gru_cell_1_222182*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_2218352$
"gru_cell_1/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_222178gru_cell_1_222180gru_cell_1_222182*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_222190*
condR
while_cond_222189*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^gru_cell_1/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????	
 
_user_specified_nameinputs
?h
?
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_221931

inputs

states
readvariableop_resource
readvariableop_1_resource
readvariableop_4_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6X
ones_like/ShapeShapestates*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_likey
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slicem
MatMulMatMulinputsstrided_slice:output:0*
T0*(
_output_shapes
:??????????2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slice_1s
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slice_2s
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack}
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
strided_slice_3|
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddy
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_4/stack}
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_1y
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_2`
mulMulstatesones_like:output:0*
T0*(
_output_shapes
:??????????2
muld
mul_1Mulstatesones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_1d
mul_2Mulstatesones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_2?
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_6t
MatMul_3MatMulmul:z:0strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2

MatMul_3?
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_7v
MatMul_4MatMul	mul_1:z:0strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack}
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
strided_slice_8?
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_3y
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_9/stack}
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
strided_slice_9?
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_4l
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidr
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1?
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_6?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_10w
MatMul_5MatMul	mul_2:z:0strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5{
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
strided_slice_11?
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_5k
mul_3MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
mul_3i
add_2AddV2BiasAdd_2:output:0	mul_3:z:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh]
mul_4MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
mul_4S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sub[
mul_5Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_5`
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:??????????2
add_3?
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????	:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_6:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
??
?
while_body_224350
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_00
,while_gru_cell_1_readvariableop_1_resource_00
,while_gru_cell_1_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource.
*while_gru_cell_1_readvariableop_1_resource.
*while_gru_cell_1_readvariableop_4_resource??while/gru_cell_1/ReadVariableOp?!while/gru_cell_1/ReadVariableOp_1?!while/gru_cell_1/ReadVariableOp_2?!while/gru_cell_1/ReadVariableOp_3?!while/gru_cell_1/ReadVariableOp_4?!while/gru_cell_1/ReadVariableOp_5?!while/gru_cell_1/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_1/ones_like/Shape?
 while/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_1/ones_like/Const?
while/gru_cell_1/ones_likeFill)while/gru_cell_1/ones_like/Shape:output:0)while/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/ones_like?
while/gru_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell_1/dropout/Const?
while/gru_cell_1/dropout/MulMul#while/gru_cell_1/ones_like:output:0'while/gru_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/dropout/Mul?
while/gru_cell_1/dropout/ShapeShape#while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell_1/dropout/Shape?
5while/gru_cell_1/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ը27
5while/gru_cell_1/dropout/random_uniform/RandomUniform?
'while/gru_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell_1/dropout/GreaterEqual/y?
%while/gru_cell_1/dropout/GreaterEqualGreaterEqual>while/gru_cell_1/dropout/random_uniform/RandomUniform:output:00while/gru_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell_1/dropout/GreaterEqual?
while/gru_cell_1/dropout/CastCast)while/gru_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell_1/dropout/Cast?
while/gru_cell_1/dropout/Mul_1Mul while/gru_cell_1/dropout/Mul:z:0!while/gru_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_1/dropout/Mul_1?
 while/gru_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_1/dropout_1/Const?
while/gru_cell_1/dropout_1/MulMul#while/gru_cell_1/ones_like:output:0)while/gru_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_1/dropout_1/Mul?
 while/gru_cell_1/dropout_1/ShapeShape#while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_1/dropout_1/Shape?
7while/gru_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???29
7while/gru_cell_1/dropout_1/random_uniform/RandomUniform?
)while/gru_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_1/dropout_1/GreaterEqual/y?
'while/gru_cell_1/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_1/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_1/dropout_1/GreaterEqual?
while/gru_cell_1/dropout_1/CastCast+while/gru_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_1/dropout_1/Cast?
 while/gru_cell_1/dropout_1/Mul_1Mul"while/gru_cell_1/dropout_1/Mul:z:0#while/gru_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_1/dropout_1/Mul_1?
 while/gru_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_1/dropout_2/Const?
while/gru_cell_1/dropout_2/MulMul#while/gru_cell_1/ones_like:output:0)while/gru_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_1/dropout_2/Mul?
 while/gru_cell_1/dropout_2/ShapeShape#while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_1/dropout_2/Shape?
7while/gru_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ò?29
7while/gru_cell_1/dropout_2/random_uniform/RandomUniform?
)while/gru_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_1/dropout_2/GreaterEqual/y?
'while/gru_cell_1/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_1/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_1/dropout_2/GreaterEqual?
while/gru_cell_1/dropout_2/CastCast+while/gru_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_1/dropout_2/Cast?
 while/gru_cell_1/dropout_2/Mul_1Mul"while/gru_cell_1/dropout_2/Mul:z:0#while/gru_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_1/dropout_2/Mul_1?
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_1/ReadVariableOp?
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_1/unstack?
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_1?
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_1/strided_slice/stack?
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice/stack_1?
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_1/strided_slice/stack_2?
while/gru_cell_1/strided_sliceStridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_1/strided_slice?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul?
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_2?
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_1/stack?
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_1/stack_1?
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_1/stack_2?
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_1?
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_1?
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_3?
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_1/strided_slice_2/stack?
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_1/strided_slice_2/stack_1?
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_2/stack_2?
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_2?
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_2?
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_3/stack?
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_3/stack_1?
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_3/stack_2?
 while/gru_cell_1/strided_slice_3StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_3?
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd?
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_4/stack?
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_4/stack_1?
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_4/stack_2?
 while/gru_cell_1/strided_slice_4StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_4?
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_1?
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_5/stack?
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_1/strided_slice_5/stack_1?
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_5/stack_2?
 while/gru_cell_1/strided_slice_5StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_1/strided_slice_5?
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_2?
while/gru_cell_1/mulMulwhile_placeholder_2"while/gru_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul?
while/gru_cell_1/mul_1Mulwhile_placeholder_2$while/gru_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_1?
while/gru_cell_1/mul_2Mulwhile_placeholder_2$while/gru_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_2?
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_4?
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_1/strided_slice_6/stack?
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_1/strided_slice_6/stack_1?
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_6/stack_2?
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_6?
while/gru_cell_1/MatMul_3MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_3?
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_5?
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_7/stack?
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_7/stack_1?
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_7/stack_2?
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_7?
while/gru_cell_1/MatMul_4MatMulwhile/gru_cell_1/mul_1:z:0)while/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_4?
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_8/stack?
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_8/stack_1?
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_8/stack_2?
 while/gru_cell_1/strided_slice_8StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_8?
while/gru_cell_1/BiasAdd_3BiasAdd#while/gru_cell_1/MatMul_3:product:0)while/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_3?
&while/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_9/stack?
(while/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_9/stack_1?
(while/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_9/stack_2?
 while/gru_cell_1/strided_slice_9StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_9/stack:output:01while/gru_cell_1/strided_slice_9/stack_1:output:01while/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_9?
while/gru_cell_1/BiasAdd_4BiasAdd#while/gru_cell_1/MatMul_4:product:0)while/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_4?
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add?
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid?
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_1?
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid_1?
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_6?
'while/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_1/strided_slice_10/stack?
)while/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_1/strided_slice_10/stack_1?
)while/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_1/strided_slice_10/stack_2?
!while/gru_cell_1/strided_slice_10StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:00while/gru_cell_1/strided_slice_10/stack:output:02while/gru_cell_1/strided_slice_10/stack_1:output:02while/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_1/strided_slice_10?
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul_2:z:0*while/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_5?
'while/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_1/strided_slice_11/stack?
)while/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_1/strided_slice_11/stack_1?
)while/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_1/strided_slice_11/stack_2?
!while/gru_cell_1/strided_slice_11StridedSlice!while/gru_cell_1/unstack:output:10while/gru_cell_1/strided_slice_11/stack:output:02while/gru_cell_1/strided_slice_11/stack_1:output:02while/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_1/strided_slice_11?
while/gru_cell_1/BiasAdd_5BiasAdd#while/gru_cell_1/MatMul_5:product:0*while/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_5?
while/gru_cell_1/mul_3Mulwhile/gru_cell_1/Sigmoid_1:y:0#while/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_3?
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0while/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_2?
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Tanh?
while/gru_cell_1/mul_4Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_4u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_1/sub/x?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/sub?
while/gru_cell_1/mul_5Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_5?
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_4:z:0while/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_1_readvariableop_1_resource,while_gru_cell_1_readvariableop_1_resource_0"Z
*while_gru_cell_1_readvariableop_4_resource,while_gru_cell_1_readvariableop_4_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
model_1_gru_1_while_cond_2214688
4model_1_gru_1_while_model_1_gru_1_while_loop_counter>
:model_1_gru_1_while_model_1_gru_1_while_maximum_iterations#
model_1_gru_1_while_placeholder%
!model_1_gru_1_while_placeholder_1%
!model_1_gru_1_while_placeholder_2:
6model_1_gru_1_while_less_model_1_gru_1_strided_slice_1P
Lmodel_1_gru_1_while_model_1_gru_1_while_cond_221468___redundant_placeholder0P
Lmodel_1_gru_1_while_model_1_gru_1_while_cond_221468___redundant_placeholder1P
Lmodel_1_gru_1_while_model_1_gru_1_while_cond_221468___redundant_placeholder2P
Lmodel_1_gru_1_while_model_1_gru_1_while_cond_221468___redundant_placeholder3 
model_1_gru_1_while_identity
?
model_1/gru_1/while/LessLessmodel_1_gru_1_while_placeholder6model_1_gru_1_while_less_model_1_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
model_1/gru_1/while/Less?
model_1/gru_1/while/IdentityIdentitymodel_1/gru_1/while/Less:z:0*
T0
*
_output_shapes
: 2
model_1/gru_1/while/Identity"E
model_1_gru_1_while_identity%model_1/gru_1/while/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
}
(__inference_dense_4_layer_call_fn_225496

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2230742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_gru_1_layer_call_fn_224813
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_2223722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????	
"
_user_specified_name
inputs/0
??
?
A__inference_gru_1_layer_call_and_return_conditional_losses_224520
inputs_0&
"gru_cell_1_readvariableop_resource(
$gru_cell_1_readvariableop_1_resource(
$gru_cell_1_readvariableop_4_resource
identity??gru_cell_1/ReadVariableOp?gru_cell_1/ReadVariableOp_1?gru_cell_1/ReadVariableOp_2?gru_cell_1/ReadVariableOp_3?gru_cell_1/ReadVariableOp_4?gru_cell_1/ReadVariableOp_5?gru_cell_1/ReadVariableOp_6?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????	2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
strided_slice_2v
gru_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_1/ones_like/Shape}
gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/ones_like/Const?
gru_cell_1/ones_likeFill#gru_cell_1/ones_like/Shape:output:0#gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/ones_likey
gru_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_1/dropout/Const?
gru_cell_1/dropout/MulMulgru_cell_1/ones_like:output:0!gru_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout/Mul?
gru_cell_1/dropout/ShapeShapegru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_1/dropout/Shape?
/gru_cell_1/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??521
/gru_cell_1/dropout/random_uniform/RandomUniform?
!gru_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell_1/dropout/GreaterEqual/y?
gru_cell_1/dropout/GreaterEqualGreaterEqual8gru_cell_1/dropout/random_uniform/RandomUniform:output:0*gru_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell_1/dropout/GreaterEqual?
gru_cell_1/dropout/CastCast#gru_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_1/dropout/Cast?
gru_cell_1/dropout/Mul_1Mulgru_cell_1/dropout/Mul:z:0gru_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout/Mul_1}
gru_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_1/dropout_1/Const?
gru_cell_1/dropout_1/MulMulgru_cell_1/ones_like:output:0#gru_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_1/Mul?
gru_cell_1/dropout_1/ShapeShapegru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_1/dropout_1/Shape?
1gru_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??23
1gru_cell_1/dropout_1/random_uniform/RandomUniform?
#gru_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_1/dropout_1/GreaterEqual/y?
!gru_cell_1/dropout_1/GreaterEqualGreaterEqual:gru_cell_1/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_1/dropout_1/GreaterEqual?
gru_cell_1/dropout_1/CastCast%gru_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_1/dropout_1/Cast?
gru_cell_1/dropout_1/Mul_1Mulgru_cell_1/dropout_1/Mul:z:0gru_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_1/Mul_1}
gru_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_1/dropout_2/Const?
gru_cell_1/dropout_2/MulMulgru_cell_1/ones_like:output:0#gru_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_2/Mul?
gru_cell_1/dropout_2/ShapeShapegru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_1/dropout_2/Shape?
1gru_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???23
1gru_cell_1/dropout_2/random_uniform/RandomUniform?
#gru_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_1/dropout_2/GreaterEqual/y?
!gru_cell_1/dropout_2/GreaterEqualGreaterEqual:gru_cell_1/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_1/dropout_2/GreaterEqual?
gru_cell_1/dropout_2/CastCast%gru_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_1/dropout_2/Cast?
gru_cell_1/dropout_2/Mul_1Mulgru_cell_1/dropout_2/Mul:z:0gru_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_2/Mul_1?
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_1/ReadVariableOp?
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_1/unstack?
gru_cell_1/ReadVariableOp_1ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_1?
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_1/strided_slice/stack?
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice/stack_1?
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_1/strided_slice/stack_2?
gru_cell_1/strided_sliceStridedSlice#gru_cell_1/ReadVariableOp_1:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0!gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul?
gru_cell_1/ReadVariableOp_2ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_2?
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_1/stack?
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_1/stack_1?
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_1/stack_2?
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_1?
gru_cell_1/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_1?
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_3?
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_1/strided_slice_2/stack?
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_1/strided_slice_2/stack_1?
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_2/stack_2?
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_2?
gru_cell_1/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_2?
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_3/stack?
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_3/stack_1?
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_3/stack_2?
gru_cell_1/strided_slice_3StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_3?
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd?
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_4/stack?
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_4/stack_1?
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_4/stack_2?
gru_cell_1/strided_slice_4StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_4?
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_1?
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_5/stack?
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_1/strided_slice_5/stack_1?
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_5/stack_2?
gru_cell_1/strided_slice_5StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_5?
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_2?
gru_cell_1/mulMulzeros:output:0gru_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul?
gru_cell_1/mul_1Mulzeros:output:0gru_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_1?
gru_cell_1/mul_2Mulzeros:output:0gru_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_2?
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_4?
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_1/strided_slice_6/stack?
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_1/strided_slice_6/stack_1?
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_6/stack_2?
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_6?
gru_cell_1/MatMul_3MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_3?
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_5?
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_7/stack?
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_7/stack_1?
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_7/stack_2?
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_7?
gru_cell_1/MatMul_4MatMulgru_cell_1/mul_1:z:0#gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_4?
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_8/stack?
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_8/stack_1?
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_8/stack_2?
gru_cell_1/strided_slice_8StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_8?
gru_cell_1/BiasAdd_3BiasAddgru_cell_1/MatMul_3:product:0#gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_3?
 gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_9/stack?
"gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_9/stack_1?
"gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_9/stack_2?
gru_cell_1/strided_slice_9StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_9/stack:output:0+gru_cell_1/strided_slice_9/stack_1:output:0+gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_9?
gru_cell_1/BiasAdd_4BiasAddgru_cell_1/MatMul_4:product:0#gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_4?
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/addz
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid?
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_1?
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid_1?
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_6?
!gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_1/strided_slice_10/stack?
#gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_1/strided_slice_10/stack_1?
#gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_1/strided_slice_10/stack_2?
gru_cell_1/strided_slice_10StridedSlice#gru_cell_1/ReadVariableOp_6:value:0*gru_cell_1/strided_slice_10/stack:output:0,gru_cell_1/strided_slice_10/stack_1:output:0,gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_10?
gru_cell_1/MatMul_5MatMulgru_cell_1/mul_2:z:0$gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_5?
!gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_1/strided_slice_11/stack?
#gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_1/strided_slice_11/stack_1?
#gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_1/strided_slice_11/stack_2?
gru_cell_1/strided_slice_11StridedSlicegru_cell_1/unstack:output:1*gru_cell_1/strided_slice_11/stack:output:0,gru_cell_1/strided_slice_11/stack_1:output:0,gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_11?
gru_cell_1/BiasAdd_5BiasAddgru_cell_1/MatMul_5:product:0$gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_5?
gru_cell_1/mul_3Mulgru_cell_1/Sigmoid_1:y:0gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_3?
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_2s
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Tanh?
gru_cell_1/mul_4Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_4i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/sub/x?
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/sub?
gru_cell_1/mul_5Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_5?
gru_cell_1/add_3AddV2gru_cell_1/mul_4:z:0gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_1_resource$gru_cell_1_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_224350*
condR
while_cond_224349*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62
whilewhile:^ Z
4
_output_shapes"
 :??????????????????	
"
_user_specified_name
inputs/0
?
}
(__inference_dense_7_layer_call_fn_225569

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2231712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?<
?
A__inference_gru_1_layer_call_and_return_conditional_losses_222372

inputs
gru_cell_1_222296
gru_cell_1_222298
gru_cell_1_222300
identity??"gru_cell_1/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????	2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
strided_slice_2?
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_1_222296gru_cell_1_222298gru_cell_1_222300*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_2219312$
"gru_cell_1/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_222296gru_cell_1_222298gru_cell_1_222300*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_222308*
condR
while_cond_222307*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^gru_cell_1/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????	
 
_user_specified_nameinputs
??
?
model_1_gru_1_while_body_2214698
4model_1_gru_1_while_model_1_gru_1_while_loop_counter>
:model_1_gru_1_while_model_1_gru_1_while_maximum_iterations#
model_1_gru_1_while_placeholder%
!model_1_gru_1_while_placeholder_1%
!model_1_gru_1_while_placeholder_27
3model_1_gru_1_while_model_1_gru_1_strided_slice_1_0s
omodel_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_1_gru_1_tensorarrayunstack_tensorlistfromtensor_0<
8model_1_gru_1_while_gru_cell_1_readvariableop_resource_0>
:model_1_gru_1_while_gru_cell_1_readvariableop_1_resource_0>
:model_1_gru_1_while_gru_cell_1_readvariableop_4_resource_0 
model_1_gru_1_while_identity"
model_1_gru_1_while_identity_1"
model_1_gru_1_while_identity_2"
model_1_gru_1_while_identity_3"
model_1_gru_1_while_identity_45
1model_1_gru_1_while_model_1_gru_1_strided_slice_1q
mmodel_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_1_gru_1_tensorarrayunstack_tensorlistfromtensor:
6model_1_gru_1_while_gru_cell_1_readvariableop_resource<
8model_1_gru_1_while_gru_cell_1_readvariableop_1_resource<
8model_1_gru_1_while_gru_cell_1_readvariableop_4_resource??-model_1/gru_1/while/gru_cell_1/ReadVariableOp?/model_1/gru_1/while/gru_cell_1/ReadVariableOp_1?/model_1/gru_1/while/gru_cell_1/ReadVariableOp_2?/model_1/gru_1/while/gru_cell_1/ReadVariableOp_3?/model_1/gru_1/while/gru_cell_1/ReadVariableOp_4?/model_1/gru_1/while/gru_cell_1/ReadVariableOp_5?/model_1/gru_1/while/gru_cell_1/ReadVariableOp_6?
Emodel_1/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2G
Emodel_1/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
7model_1/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemomodel_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_1_gru_1_tensorarrayunstack_tensorlistfromtensor_0model_1_gru_1_while_placeholderNmodel_1/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype029
7model_1/gru_1/while/TensorArrayV2Read/TensorListGetItem?
.model_1/gru_1/while/gru_cell_1/ones_like/ShapeShape!model_1_gru_1_while_placeholder_2*
T0*
_output_shapes
:20
.model_1/gru_1/while/gru_cell_1/ones_like/Shape?
.model_1/gru_1/while/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.model_1/gru_1/while/gru_cell_1/ones_like/Const?
(model_1/gru_1/while/gru_cell_1/ones_likeFill7model_1/gru_1/while/gru_cell_1/ones_like/Shape:output:07model_1/gru_1/while/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2*
(model_1/gru_1/while/gru_cell_1/ones_like?
-model_1/gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp8model_1_gru_1_while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-model_1/gru_1/while/gru_cell_1/ReadVariableOp?
&model_1/gru_1/while/gru_cell_1/unstackUnpack5model_1/gru_1/while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2(
&model_1/gru_1/while/gru_cell_1/unstack?
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_1ReadVariableOp:model_1_gru_1_while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype021
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_1?
2model_1/gru_1/while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2model_1/gru_1/while/gru_cell_1/strided_slice/stack?
4model_1/gru_1/while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   26
4model_1/gru_1/while/gru_cell_1/strided_slice/stack_1?
4model_1/gru_1/while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model_1/gru_1/while/gru_cell_1/strided_slice/stack_2?
,model_1/gru_1/while/gru_cell_1/strided_sliceStridedSlice7model_1/gru_1/while/gru_cell_1/ReadVariableOp_1:value:0;model_1/gru_1/while/gru_cell_1/strided_slice/stack:output:0=model_1/gru_1/while/gru_cell_1/strided_slice/stack_1:output:0=model_1/gru_1/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2.
,model_1/gru_1/while/gru_cell_1/strided_slice?
%model_1/gru_1/while/gru_cell_1/MatMulMatMul>model_1/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:05model_1/gru_1/while/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2'
%model_1/gru_1/while/gru_cell_1/MatMul?
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_2ReadVariableOp:model_1_gru_1_while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype021
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_2?
4model_1/gru_1/while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   26
4model_1/gru_1/while/gru_cell_1/strided_slice_1/stack?
6model_1/gru_1/while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  28
6model_1/gru_1/while/gru_cell_1/strided_slice_1/stack_1?
6model_1/gru_1/while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6model_1/gru_1/while/gru_cell_1/strided_slice_1/stack_2?
.model_1/gru_1/while/gru_cell_1/strided_slice_1StridedSlice7model_1/gru_1/while/gru_cell_1/ReadVariableOp_2:value:0=model_1/gru_1/while/gru_cell_1/strided_slice_1/stack:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_1/stack_1:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask20
.model_1/gru_1/while/gru_cell_1/strided_slice_1?
'model_1/gru_1/while/gru_cell_1/MatMul_1MatMul>model_1/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:07model_1/gru_1/while/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2)
'model_1/gru_1/while/gru_cell_1/MatMul_1?
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_3ReadVariableOp:model_1_gru_1_while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype021
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_3?
4model_1/gru_1/while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  26
4model_1/gru_1/while/gru_cell_1/strided_slice_2/stack?
6model_1/gru_1/while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        28
6model_1/gru_1/while/gru_cell_1/strided_slice_2/stack_1?
6model_1/gru_1/while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6model_1/gru_1/while/gru_cell_1/strided_slice_2/stack_2?
.model_1/gru_1/while/gru_cell_1/strided_slice_2StridedSlice7model_1/gru_1/while/gru_cell_1/ReadVariableOp_3:value:0=model_1/gru_1/while/gru_cell_1/strided_slice_2/stack:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_2/stack_1:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask20
.model_1/gru_1/while/gru_cell_1/strided_slice_2?
'model_1/gru_1/while/gru_cell_1/MatMul_2MatMul>model_1/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:07model_1/gru_1/while/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2)
'model_1/gru_1/while/gru_cell_1/MatMul_2?
4model_1/gru_1/while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4model_1/gru_1/while/gru_cell_1/strided_slice_3/stack?
6model_1/gru_1/while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?28
6model_1/gru_1/while/gru_cell_1/strided_slice_3/stack_1?
6model_1/gru_1/while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model_1/gru_1/while/gru_cell_1/strided_slice_3/stack_2?
.model_1/gru_1/while/gru_cell_1/strided_slice_3StridedSlice/model_1/gru_1/while/gru_cell_1/unstack:output:0=model_1/gru_1/while/gru_cell_1/strided_slice_3/stack:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_3/stack_1:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask20
.model_1/gru_1/while/gru_cell_1/strided_slice_3?
&model_1/gru_1/while/gru_cell_1/BiasAddBiasAdd/model_1/gru_1/while/gru_cell_1/MatMul:product:07model_1/gru_1/while/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2(
&model_1/gru_1/while/gru_cell_1/BiasAdd?
4model_1/gru_1/while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?26
4model_1/gru_1/while/gru_cell_1/strided_slice_4/stack?
6model_1/gru_1/while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?28
6model_1/gru_1/while/gru_cell_1/strided_slice_4/stack_1?
6model_1/gru_1/while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model_1/gru_1/while/gru_cell_1/strided_slice_4/stack_2?
.model_1/gru_1/while/gru_cell_1/strided_slice_4StridedSlice/model_1/gru_1/while/gru_cell_1/unstack:output:0=model_1/gru_1/while/gru_cell_1/strided_slice_4/stack:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_4/stack_1:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?20
.model_1/gru_1/while/gru_cell_1/strided_slice_4?
(model_1/gru_1/while/gru_cell_1/BiasAdd_1BiasAdd1model_1/gru_1/while/gru_cell_1/MatMul_1:product:07model_1/gru_1/while/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2*
(model_1/gru_1/while/gru_cell_1/BiasAdd_1?
4model_1/gru_1/while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?26
4model_1/gru_1/while/gru_cell_1/strided_slice_5/stack?
6model_1/gru_1/while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6model_1/gru_1/while/gru_cell_1/strided_slice_5/stack_1?
6model_1/gru_1/while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model_1/gru_1/while/gru_cell_1/strided_slice_5/stack_2?
.model_1/gru_1/while/gru_cell_1/strided_slice_5StridedSlice/model_1/gru_1/while/gru_cell_1/unstack:output:0=model_1/gru_1/while/gru_cell_1/strided_slice_5/stack:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_5/stack_1:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask20
.model_1/gru_1/while/gru_cell_1/strided_slice_5?
(model_1/gru_1/while/gru_cell_1/BiasAdd_2BiasAdd1model_1/gru_1/while/gru_cell_1/MatMul_2:product:07model_1/gru_1/while/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2*
(model_1/gru_1/while/gru_cell_1/BiasAdd_2?
"model_1/gru_1/while/gru_cell_1/mulMul!model_1_gru_1_while_placeholder_21model_1/gru_1/while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2$
"model_1/gru_1/while/gru_cell_1/mul?
$model_1/gru_1/while/gru_cell_1/mul_1Mul!model_1_gru_1_while_placeholder_21model_1/gru_1/while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2&
$model_1/gru_1/while/gru_cell_1/mul_1?
$model_1/gru_1/while/gru_cell_1/mul_2Mul!model_1_gru_1_while_placeholder_21model_1/gru_1/while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2&
$model_1/gru_1/while/gru_cell_1/mul_2?
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_4ReadVariableOp:model_1_gru_1_while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype021
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_4?
4model_1/gru_1/while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4model_1/gru_1/while/gru_cell_1/strided_slice_6/stack?
6model_1/gru_1/while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   28
6model_1/gru_1/while/gru_cell_1/strided_slice_6/stack_1?
6model_1/gru_1/while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6model_1/gru_1/while/gru_cell_1/strided_slice_6/stack_2?
.model_1/gru_1/while/gru_cell_1/strided_slice_6StridedSlice7model_1/gru_1/while/gru_cell_1/ReadVariableOp_4:value:0=model_1/gru_1/while/gru_cell_1/strided_slice_6/stack:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_6/stack_1:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask20
.model_1/gru_1/while/gru_cell_1/strided_slice_6?
'model_1/gru_1/while/gru_cell_1/MatMul_3MatMul&model_1/gru_1/while/gru_cell_1/mul:z:07model_1/gru_1/while/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2)
'model_1/gru_1/while/gru_cell_1/MatMul_3?
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_5ReadVariableOp:model_1_gru_1_while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype021
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_5?
4model_1/gru_1/while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   26
4model_1/gru_1/while/gru_cell_1/strided_slice_7/stack?
6model_1/gru_1/while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  28
6model_1/gru_1/while/gru_cell_1/strided_slice_7/stack_1?
6model_1/gru_1/while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6model_1/gru_1/while/gru_cell_1/strided_slice_7/stack_2?
.model_1/gru_1/while/gru_cell_1/strided_slice_7StridedSlice7model_1/gru_1/while/gru_cell_1/ReadVariableOp_5:value:0=model_1/gru_1/while/gru_cell_1/strided_slice_7/stack:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_7/stack_1:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask20
.model_1/gru_1/while/gru_cell_1/strided_slice_7?
'model_1/gru_1/while/gru_cell_1/MatMul_4MatMul(model_1/gru_1/while/gru_cell_1/mul_1:z:07model_1/gru_1/while/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2)
'model_1/gru_1/while/gru_cell_1/MatMul_4?
4model_1/gru_1/while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4model_1/gru_1/while/gru_cell_1/strided_slice_8/stack?
6model_1/gru_1/while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?28
6model_1/gru_1/while/gru_cell_1/strided_slice_8/stack_1?
6model_1/gru_1/while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model_1/gru_1/while/gru_cell_1/strided_slice_8/stack_2?
.model_1/gru_1/while/gru_cell_1/strided_slice_8StridedSlice/model_1/gru_1/while/gru_cell_1/unstack:output:1=model_1/gru_1/while/gru_cell_1/strided_slice_8/stack:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_8/stack_1:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask20
.model_1/gru_1/while/gru_cell_1/strided_slice_8?
(model_1/gru_1/while/gru_cell_1/BiasAdd_3BiasAdd1model_1/gru_1/while/gru_cell_1/MatMul_3:product:07model_1/gru_1/while/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2*
(model_1/gru_1/while/gru_cell_1/BiasAdd_3?
4model_1/gru_1/while/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?26
4model_1/gru_1/while/gru_cell_1/strided_slice_9/stack?
6model_1/gru_1/while/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?28
6model_1/gru_1/while/gru_cell_1/strided_slice_9/stack_1?
6model_1/gru_1/while/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model_1/gru_1/while/gru_cell_1/strided_slice_9/stack_2?
.model_1/gru_1/while/gru_cell_1/strided_slice_9StridedSlice/model_1/gru_1/while/gru_cell_1/unstack:output:1=model_1/gru_1/while/gru_cell_1/strided_slice_9/stack:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_9/stack_1:output:0?model_1/gru_1/while/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?20
.model_1/gru_1/while/gru_cell_1/strided_slice_9?
(model_1/gru_1/while/gru_cell_1/BiasAdd_4BiasAdd1model_1/gru_1/while/gru_cell_1/MatMul_4:product:07model_1/gru_1/while/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2*
(model_1/gru_1/while/gru_cell_1/BiasAdd_4?
"model_1/gru_1/while/gru_cell_1/addAddV2/model_1/gru_1/while/gru_cell_1/BiasAdd:output:01model_1/gru_1/while/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2$
"model_1/gru_1/while/gru_cell_1/add?
&model_1/gru_1/while/gru_cell_1/SigmoidSigmoid&model_1/gru_1/while/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2(
&model_1/gru_1/while/gru_cell_1/Sigmoid?
$model_1/gru_1/while/gru_cell_1/add_1AddV21model_1/gru_1/while/gru_cell_1/BiasAdd_1:output:01model_1/gru_1/while/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2&
$model_1/gru_1/while/gru_cell_1/add_1?
(model_1/gru_1/while/gru_cell_1/Sigmoid_1Sigmoid(model_1/gru_1/while/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2*
(model_1/gru_1/while/gru_cell_1/Sigmoid_1?
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_6ReadVariableOp:model_1_gru_1_while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype021
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_6?
5model_1/gru_1/while/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  27
5model_1/gru_1/while/gru_cell_1/strided_slice_10/stack?
7model_1/gru_1/while/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7model_1/gru_1/while/gru_cell_1/strided_slice_10/stack_1?
7model_1/gru_1/while/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7model_1/gru_1/while/gru_cell_1/strided_slice_10/stack_2?
/model_1/gru_1/while/gru_cell_1/strided_slice_10StridedSlice7model_1/gru_1/while/gru_cell_1/ReadVariableOp_6:value:0>model_1/gru_1/while/gru_cell_1/strided_slice_10/stack:output:0@model_1/gru_1/while/gru_cell_1/strided_slice_10/stack_1:output:0@model_1/gru_1/while/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask21
/model_1/gru_1/while/gru_cell_1/strided_slice_10?
'model_1/gru_1/while/gru_cell_1/MatMul_5MatMul(model_1/gru_1/while/gru_cell_1/mul_2:z:08model_1/gru_1/while/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2)
'model_1/gru_1/while/gru_cell_1/MatMul_5?
5model_1/gru_1/while/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?27
5model_1/gru_1/while/gru_cell_1/strided_slice_11/stack?
7model_1/gru_1/while/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7model_1/gru_1/while/gru_cell_1/strided_slice_11/stack_1?
7model_1/gru_1/while/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_1/gru_1/while/gru_cell_1/strided_slice_11/stack_2?
/model_1/gru_1/while/gru_cell_1/strided_slice_11StridedSlice/model_1/gru_1/while/gru_cell_1/unstack:output:1>model_1/gru_1/while/gru_cell_1/strided_slice_11/stack:output:0@model_1/gru_1/while/gru_cell_1/strided_slice_11/stack_1:output:0@model_1/gru_1/while/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask21
/model_1/gru_1/while/gru_cell_1/strided_slice_11?
(model_1/gru_1/while/gru_cell_1/BiasAdd_5BiasAdd1model_1/gru_1/while/gru_cell_1/MatMul_5:product:08model_1/gru_1/while/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2*
(model_1/gru_1/while/gru_cell_1/BiasAdd_5?
$model_1/gru_1/while/gru_cell_1/mul_3Mul,model_1/gru_1/while/gru_cell_1/Sigmoid_1:y:01model_1/gru_1/while/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2&
$model_1/gru_1/while/gru_cell_1/mul_3?
$model_1/gru_1/while/gru_cell_1/add_2AddV21model_1/gru_1/while/gru_cell_1/BiasAdd_2:output:0(model_1/gru_1/while/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2&
$model_1/gru_1/while/gru_cell_1/add_2?
#model_1/gru_1/while/gru_cell_1/TanhTanh(model_1/gru_1/while/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2%
#model_1/gru_1/while/gru_cell_1/Tanh?
$model_1/gru_1/while/gru_cell_1/mul_4Mul*model_1/gru_1/while/gru_cell_1/Sigmoid:y:0!model_1_gru_1_while_placeholder_2*
T0*(
_output_shapes
:??????????2&
$model_1/gru_1/while/gru_cell_1/mul_4?
$model_1/gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$model_1/gru_1/while/gru_cell_1/sub/x?
"model_1/gru_1/while/gru_cell_1/subSub-model_1/gru_1/while/gru_cell_1/sub/x:output:0*model_1/gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2$
"model_1/gru_1/while/gru_cell_1/sub?
$model_1/gru_1/while/gru_cell_1/mul_5Mul&model_1/gru_1/while/gru_cell_1/sub:z:0'model_1/gru_1/while/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2&
$model_1/gru_1/while/gru_cell_1/mul_5?
$model_1/gru_1/while/gru_cell_1/add_3AddV2(model_1/gru_1/while/gru_cell_1/mul_4:z:0(model_1/gru_1/while/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2&
$model_1/gru_1/while/gru_cell_1/add_3?
8model_1/gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!model_1_gru_1_while_placeholder_1model_1_gru_1_while_placeholder(model_1/gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02:
8model_1/gru_1/while/TensorArrayV2Write/TensorListSetItemx
model_1/gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_1/gru_1/while/add/y?
model_1/gru_1/while/addAddV2model_1_gru_1_while_placeholder"model_1/gru_1/while/add/y:output:0*
T0*
_output_shapes
: 2
model_1/gru_1/while/add|
model_1/gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_1/gru_1/while/add_1/y?
model_1/gru_1/while/add_1AddV24model_1_gru_1_while_model_1_gru_1_while_loop_counter$model_1/gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_1/gru_1/while/add_1?
model_1/gru_1/while/IdentityIdentitymodel_1/gru_1/while/add_1:z:0.^model_1/gru_1/while/gru_cell_1/ReadVariableOp0^model_1/gru_1/while/gru_cell_1/ReadVariableOp_10^model_1/gru_1/while/gru_cell_1/ReadVariableOp_20^model_1/gru_1/while/gru_cell_1/ReadVariableOp_30^model_1/gru_1/while/gru_cell_1/ReadVariableOp_40^model_1/gru_1/while/gru_cell_1/ReadVariableOp_50^model_1/gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
model_1/gru_1/while/Identity?
model_1/gru_1/while/Identity_1Identity:model_1_gru_1_while_model_1_gru_1_while_maximum_iterations.^model_1/gru_1/while/gru_cell_1/ReadVariableOp0^model_1/gru_1/while/gru_cell_1/ReadVariableOp_10^model_1/gru_1/while/gru_cell_1/ReadVariableOp_20^model_1/gru_1/while/gru_cell_1/ReadVariableOp_30^model_1/gru_1/while/gru_cell_1/ReadVariableOp_40^model_1/gru_1/while/gru_cell_1/ReadVariableOp_50^model_1/gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2 
model_1/gru_1/while/Identity_1?
model_1/gru_1/while/Identity_2Identitymodel_1/gru_1/while/add:z:0.^model_1/gru_1/while/gru_cell_1/ReadVariableOp0^model_1/gru_1/while/gru_cell_1/ReadVariableOp_10^model_1/gru_1/while/gru_cell_1/ReadVariableOp_20^model_1/gru_1/while/gru_cell_1/ReadVariableOp_30^model_1/gru_1/while/gru_cell_1/ReadVariableOp_40^model_1/gru_1/while/gru_cell_1/ReadVariableOp_50^model_1/gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2 
model_1/gru_1/while/Identity_2?
model_1/gru_1/while/Identity_3IdentityHmodel_1/gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0.^model_1/gru_1/while/gru_cell_1/ReadVariableOp0^model_1/gru_1/while/gru_cell_1/ReadVariableOp_10^model_1/gru_1/while/gru_cell_1/ReadVariableOp_20^model_1/gru_1/while/gru_cell_1/ReadVariableOp_30^model_1/gru_1/while/gru_cell_1/ReadVariableOp_40^model_1/gru_1/while/gru_cell_1/ReadVariableOp_50^model_1/gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2 
model_1/gru_1/while/Identity_3?
model_1/gru_1/while/Identity_4Identity(model_1/gru_1/while/gru_cell_1/add_3:z:0.^model_1/gru_1/while/gru_cell_1/ReadVariableOp0^model_1/gru_1/while/gru_cell_1/ReadVariableOp_10^model_1/gru_1/while/gru_cell_1/ReadVariableOp_20^model_1/gru_1/while/gru_cell_1/ReadVariableOp_30^model_1/gru_1/while/gru_cell_1/ReadVariableOp_40^model_1/gru_1/while/gru_cell_1/ReadVariableOp_50^model_1/gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2 
model_1/gru_1/while/Identity_4"v
8model_1_gru_1_while_gru_cell_1_readvariableop_1_resource:model_1_gru_1_while_gru_cell_1_readvariableop_1_resource_0"v
8model_1_gru_1_while_gru_cell_1_readvariableop_4_resource:model_1_gru_1_while_gru_cell_1_readvariableop_4_resource_0"r
6model_1_gru_1_while_gru_cell_1_readvariableop_resource8model_1_gru_1_while_gru_cell_1_readvariableop_resource_0"E
model_1_gru_1_while_identity%model_1/gru_1/while/Identity:output:0"I
model_1_gru_1_while_identity_1'model_1/gru_1/while/Identity_1:output:0"I
model_1_gru_1_while_identity_2'model_1/gru_1/while/Identity_2:output:0"I
model_1_gru_1_while_identity_3'model_1/gru_1/while/Identity_3:output:0"I
model_1_gru_1_while_identity_4'model_1/gru_1/while/Identity_4:output:0"h
1model_1_gru_1_while_model_1_gru_1_strided_slice_13model_1_gru_1_while_model_1_gru_1_strided_slice_1_0"?
mmodel_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_1_gru_1_tensorarrayunstack_tensorlistfromtensoromodel_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_model_1_gru_1_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2^
-model_1/gru_1/while/gru_cell_1/ReadVariableOp-model_1/gru_1/while/gru_cell_1/ReadVariableOp2b
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_1/model_1/gru_1/while/gru_cell_1/ReadVariableOp_12b
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_2/model_1/gru_1/while/gru_cell_1/ReadVariableOp_22b
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_3/model_1/gru_1/while/gru_cell_1/ReadVariableOp_32b
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_4/model_1/gru_1/while/gru_cell_1/ReadVariableOp_42b
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_5/model_1/gru_1/while/gru_cell_1/ReadVariableOp_52b
/model_1/gru_1/while/gru_cell_1/ReadVariableOp_6/model_1/gru_1/while/gru_cell_1/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
(__inference_model_1_layer_call_fn_223367
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2233382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
??
?	
gru_1_while_body_223923(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_04
0gru_1_while_gru_cell_1_readvariableop_resource_06
2gru_1_while_gru_cell_1_readvariableop_1_resource_06
2gru_1_while_gru_cell_1_readvariableop_4_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor2
.gru_1_while_gru_cell_1_readvariableop_resource4
0gru_1_while_gru_cell_1_readvariableop_1_resource4
0gru_1_while_gru_cell_1_readvariableop_4_resource??%gru_1/while/gru_cell_1/ReadVariableOp?'gru_1/while/gru_cell_1/ReadVariableOp_1?'gru_1/while/gru_cell_1/ReadVariableOp_2?'gru_1/while/gru_cell_1/ReadVariableOp_3?'gru_1/while/gru_cell_1/ReadVariableOp_4?'gru_1/while/gru_cell_1/ReadVariableOp_5?'gru_1/while/gru_cell_1/ReadVariableOp_6?
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2?
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype021
/gru_1/while/TensorArrayV2Read/TensorListGetItem?
&gru_1/while/gru_cell_1/ones_like/ShapeShapegru_1_while_placeholder_2*
T0*
_output_shapes
:2(
&gru_1/while/gru_cell_1/ones_like/Shape?
&gru_1/while/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&gru_1/while/gru_cell_1/ones_like/Const?
 gru_1/while/gru_cell_1/ones_likeFill/gru_1/while/gru_cell_1/ones_like/Shape:output:0/gru_1/while/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/ones_like?
%gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	?*
dtype02'
%gru_1/while/gru_cell_1/ReadVariableOp?
gru_1/while/gru_cell_1/unstackUnpack-gru_1/while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2 
gru_1/while/gru_cell_1/unstack?
'gru_1/while/gru_cell_1/ReadVariableOp_1ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_1?
*gru_1/while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*gru_1/while/gru_cell_1/strided_slice/stack?
,gru_1/while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_1/while/gru_cell_1/strided_slice/stack_1?
,gru_1/while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru_1/while/gru_cell_1/strided_slice/stack_2?
$gru_1/while/gru_cell_1/strided_sliceStridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_1:value:03gru_1/while/gru_cell_1/strided_slice/stack:output:05gru_1/while/gru_cell_1/strided_slice/stack_1:output:05gru_1/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2&
$gru_1/while/gru_cell_1/strided_slice?
gru_1/while/gru_cell_1/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0-gru_1/while/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/MatMul?
'gru_1/while/gru_cell_1/ReadVariableOp_2ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_2?
,gru_1/while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_1/while/gru_cell_1/strided_slice_1/stack?
.gru_1/while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  20
.gru_1/while/gru_cell_1/strided_slice_1/stack_1?
.gru_1/while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_1/while/gru_cell_1/strided_slice_1/stack_2?
&gru_1/while/gru_cell_1/strided_slice_1StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_2:value:05gru_1/while/gru_cell_1/strided_slice_1/stack:output:07gru_1/while/gru_cell_1/strided_slice_1/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&gru_1/while/gru_cell_1/strided_slice_1?
gru_1/while/gru_cell_1/MatMul_1MatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_1/while/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
gru_1/while/gru_cell_1/MatMul_1?
'gru_1/while/gru_cell_1/ReadVariableOp_3ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_3?
,gru_1/while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2.
,gru_1/while/gru_cell_1/strided_slice_2/stack?
.gru_1/while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.gru_1/while/gru_cell_1/strided_slice_2/stack_1?
.gru_1/while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_1/while/gru_cell_1/strided_slice_2/stack_2?
&gru_1/while/gru_cell_1/strided_slice_2StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_3:value:05gru_1/while/gru_cell_1/strided_slice_2/stack:output:07gru_1/while/gru_cell_1/strided_slice_2/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&gru_1/while/gru_cell_1/strided_slice_2?
gru_1/while/gru_cell_1/MatMul_2MatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_1/while/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
gru_1/while/gru_cell_1/MatMul_2?
,gru_1/while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,gru_1/while/gru_cell_1/strided_slice_3/stack?
.gru_1/while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_1/while/gru_cell_1/strided_slice_3/stack_1?
.gru_1/while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_1/while/gru_cell_1/strided_slice_3/stack_2?
&gru_1/while/gru_cell_1/strided_slice_3StridedSlice'gru_1/while/gru_cell_1/unstack:output:05gru_1/while/gru_cell_1/strided_slice_3/stack:output:07gru_1/while/gru_cell_1/strided_slice_3/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2(
&gru_1/while/gru_cell_1/strided_slice_3?
gru_1/while/gru_cell_1/BiasAddBiasAdd'gru_1/while/gru_cell_1/MatMul:product:0/gru_1/while/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2 
gru_1/while/gru_cell_1/BiasAdd?
,gru_1/while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_1/while/gru_cell_1/strided_slice_4/stack?
.gru_1/while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_1/while/gru_cell_1/strided_slice_4/stack_1?
.gru_1/while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_1/while/gru_cell_1/strided_slice_4/stack_2?
&gru_1/while/gru_cell_1/strided_slice_4StridedSlice'gru_1/while/gru_cell_1/unstack:output:05gru_1/while/gru_cell_1/strided_slice_4/stack:output:07gru_1/while/gru_cell_1/strided_slice_4/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2(
&gru_1/while/gru_cell_1/strided_slice_4?
 gru_1/while/gru_cell_1/BiasAdd_1BiasAdd)gru_1/while/gru_cell_1/MatMul_1:product:0/gru_1/while/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/BiasAdd_1?
,gru_1/while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_1/while/gru_cell_1/strided_slice_5/stack?
.gru_1/while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.gru_1/while/gru_cell_1/strided_slice_5/stack_1?
.gru_1/while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_1/while/gru_cell_1/strided_slice_5/stack_2?
&gru_1/while/gru_cell_1/strided_slice_5StridedSlice'gru_1/while/gru_cell_1/unstack:output:05gru_1/while/gru_cell_1/strided_slice_5/stack:output:07gru_1/while/gru_cell_1/strided_slice_5/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2(
&gru_1/while/gru_cell_1/strided_slice_5?
 gru_1/while/gru_cell_1/BiasAdd_2BiasAdd)gru_1/while/gru_cell_1/MatMul_2:product:0/gru_1/while/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/BiasAdd_2?
gru_1/while/gru_cell_1/mulMulgru_1_while_placeholder_2)gru_1/while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul?
gru_1/while/gru_cell_1/mul_1Mulgru_1_while_placeholder_2)gru_1/while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul_1?
gru_1/while/gru_cell_1/mul_2Mulgru_1_while_placeholder_2)gru_1/while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul_2?
'gru_1/while/gru_cell_1/ReadVariableOp_4ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_4?
,gru_1/while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,gru_1/while/gru_cell_1/strided_slice_6/stack?
.gru_1/while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   20
.gru_1/while/gru_cell_1/strided_slice_6/stack_1?
.gru_1/while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_1/while/gru_cell_1/strided_slice_6/stack_2?
&gru_1/while/gru_cell_1/strided_slice_6StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_4:value:05gru_1/while/gru_cell_1/strided_slice_6/stack:output:07gru_1/while/gru_cell_1/strided_slice_6/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&gru_1/while/gru_cell_1/strided_slice_6?
gru_1/while/gru_cell_1/MatMul_3MatMulgru_1/while/gru_cell_1/mul:z:0/gru_1/while/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2!
gru_1/while/gru_cell_1/MatMul_3?
'gru_1/while/gru_cell_1/ReadVariableOp_5ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_5?
,gru_1/while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_1/while/gru_cell_1/strided_slice_7/stack?
.gru_1/while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  20
.gru_1/while/gru_cell_1/strided_slice_7/stack_1?
.gru_1/while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_1/while/gru_cell_1/strided_slice_7/stack_2?
&gru_1/while/gru_cell_1/strided_slice_7StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_5:value:05gru_1/while/gru_cell_1/strided_slice_7/stack:output:07gru_1/while/gru_cell_1/strided_slice_7/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&gru_1/while/gru_cell_1/strided_slice_7?
gru_1/while/gru_cell_1/MatMul_4MatMul gru_1/while/gru_cell_1/mul_1:z:0/gru_1/while/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2!
gru_1/while/gru_cell_1/MatMul_4?
,gru_1/while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,gru_1/while/gru_cell_1/strided_slice_8/stack?
.gru_1/while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_1/while/gru_cell_1/strided_slice_8/stack_1?
.gru_1/while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_1/while/gru_cell_1/strided_slice_8/stack_2?
&gru_1/while/gru_cell_1/strided_slice_8StridedSlice'gru_1/while/gru_cell_1/unstack:output:15gru_1/while/gru_cell_1/strided_slice_8/stack:output:07gru_1/while/gru_cell_1/strided_slice_8/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2(
&gru_1/while/gru_cell_1/strided_slice_8?
 gru_1/while/gru_cell_1/BiasAdd_3BiasAdd)gru_1/while/gru_cell_1/MatMul_3:product:0/gru_1/while/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/BiasAdd_3?
,gru_1/while/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_1/while/gru_cell_1/strided_slice_9/stack?
.gru_1/while/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_1/while/gru_cell_1/strided_slice_9/stack_1?
.gru_1/while/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_1/while/gru_cell_1/strided_slice_9/stack_2?
&gru_1/while/gru_cell_1/strided_slice_9StridedSlice'gru_1/while/gru_cell_1/unstack:output:15gru_1/while/gru_cell_1/strided_slice_9/stack:output:07gru_1/while/gru_cell_1/strided_slice_9/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2(
&gru_1/while/gru_cell_1/strided_slice_9?
 gru_1/while/gru_cell_1/BiasAdd_4BiasAdd)gru_1/while/gru_cell_1/MatMul_4:product:0/gru_1/while/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/BiasAdd_4?
gru_1/while/gru_cell_1/addAddV2'gru_1/while/gru_cell_1/BiasAdd:output:0)gru_1/while/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/add?
gru_1/while/gru_cell_1/SigmoidSigmoidgru_1/while/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2 
gru_1/while/gru_cell_1/Sigmoid?
gru_1/while/gru_cell_1/add_1AddV2)gru_1/while/gru_cell_1/BiasAdd_1:output:0)gru_1/while/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/add_1?
 gru_1/while/gru_cell_1/Sigmoid_1Sigmoid gru_1/while/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/Sigmoid_1?
'gru_1/while/gru_cell_1/ReadVariableOp_6ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_6?
-gru_1/while/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2/
-gru_1/while/gru_cell_1/strided_slice_10/stack?
/gru_1/while/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/gru_1/while/gru_cell_1/strided_slice_10/stack_1?
/gru_1/while/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/gru_1/while/gru_cell_1/strided_slice_10/stack_2?
'gru_1/while/gru_cell_1/strided_slice_10StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_6:value:06gru_1/while/gru_cell_1/strided_slice_10/stack:output:08gru_1/while/gru_cell_1/strided_slice_10/stack_1:output:08gru_1/while/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2)
'gru_1/while/gru_cell_1/strided_slice_10?
gru_1/while/gru_cell_1/MatMul_5MatMul gru_1/while/gru_cell_1/mul_2:z:00gru_1/while/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2!
gru_1/while/gru_cell_1/MatMul_5?
-gru_1/while/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2/
-gru_1/while/gru_cell_1/strided_slice_11/stack?
/gru_1/while/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/gru_1/while/gru_cell_1/strided_slice_11/stack_1?
/gru_1/while/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/gru_1/while/gru_cell_1/strided_slice_11/stack_2?
'gru_1/while/gru_cell_1/strided_slice_11StridedSlice'gru_1/while/gru_cell_1/unstack:output:16gru_1/while/gru_cell_1/strided_slice_11/stack:output:08gru_1/while/gru_cell_1/strided_slice_11/stack_1:output:08gru_1/while/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2)
'gru_1/while/gru_cell_1/strided_slice_11?
 gru_1/while/gru_cell_1/BiasAdd_5BiasAdd)gru_1/while/gru_cell_1/MatMul_5:product:00gru_1/while/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/BiasAdd_5?
gru_1/while/gru_cell_1/mul_3Mul$gru_1/while/gru_cell_1/Sigmoid_1:y:0)gru_1/while/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul_3?
gru_1/while/gru_cell_1/add_2AddV2)gru_1/while/gru_cell_1/BiasAdd_2:output:0 gru_1/while/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/add_2?
gru_1/while/gru_cell_1/TanhTanh gru_1/while/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/Tanh?
gru_1/while/gru_cell_1/mul_4Mul"gru_1/while/gru_cell_1/Sigmoid:y:0gru_1_while_placeholder_2*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul_4?
gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_1/while/gru_cell_1/sub/x?
gru_1/while/gru_cell_1/subSub%gru_1/while/gru_cell_1/sub/x:output:0"gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/sub?
gru_1/while/gru_cell_1/mul_5Mulgru_1/while/gru_cell_1/sub:z:0gru_1/while/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul_5?
gru_1/while/gru_cell_1/add_3AddV2 gru_1/while/gru_cell_1/mul_4:z:0 gru_1/while/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/add_3?
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_1/while/TensorArrayV2Write/TensorListSetItemh
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/while/add/y?
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_1/while/addl
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/while/add_1/y?
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_1/while/add_1?
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_1/while/Identity?
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_1/while/Identity_1?
gru_1/while/Identity_2Identitygru_1/while/add:z:0&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_1/while/Identity_2?
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_1/while/Identity_3?
gru_1/while/Identity_4Identity gru_1/while/gru_cell_1/add_3:z:0&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
gru_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"f
0gru_1_while_gru_cell_1_readvariableop_1_resource2gru_1_while_gru_cell_1_readvariableop_1_resource_0"f
0gru_1_while_gru_cell_1_readvariableop_4_resource2gru_1_while_gru_cell_1_readvariableop_4_resource_0"b
.gru_1_while_gru_cell_1_readvariableop_resource0gru_1_while_gru_cell_1_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"?
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2N
%gru_1/while/gru_cell_1/ReadVariableOp%gru_1/while/gru_cell_1/ReadVariableOp2R
'gru_1/while/gru_cell_1/ReadVariableOp_1'gru_1/while/gru_cell_1/ReadVariableOp_12R
'gru_1/while/gru_cell_1/ReadVariableOp_2'gru_1/while/gru_cell_1/ReadVariableOp_22R
'gru_1/while/gru_cell_1/ReadVariableOp_3'gru_1/while/gru_cell_1/ReadVariableOp_32R
'gru_1/while/gru_cell_1/ReadVariableOp_4'gru_1/while/gru_cell_1/ReadVariableOp_42R
'gru_1/while/gru_cell_1/ReadVariableOp_5'gru_1/while/gru_cell_1/ReadVariableOp_52R
'gru_1/while/gru_cell_1/ReadVariableOp_6'gru_1/while/gru_cell_1/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
(__inference_model_1_layer_call_fn_223297
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2232682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
?	
?
C__inference_dense_6_layer_call_and_return_conditional_losses_225540

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Selu?
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_5_layer_call_and_return_conditional_losses_223117

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Selu?
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_226153
file_prefix0
,assignvariableop_layer_normalization_1_gamma1
-assignvariableop_1_layer_normalization_1_beta%
!assignvariableop_2_dense_4_kernel#
assignvariableop_3_dense_4_bias%
!assignvariableop_4_dense_5_kernel#
assignvariableop_5_dense_5_bias%
!assignvariableop_6_dense_6_kernel#
assignvariableop_7_dense_6_bias%
!assignvariableop_8_dense_7_kernel#
assignvariableop_9_dense_7_bias"
assignvariableop_10_nadam_iter$
 assignvariableop_11_nadam_beta_1$
 assignvariableop_12_nadam_beta_2#
assignvariableop_13_nadam_decay+
'assignvariableop_14_nadam_learning_rate,
(assignvariableop_15_nadam_momentum_cache/
+assignvariableop_16_gru_1_gru_cell_1_kernel9
5assignvariableop_17_gru_1_gru_cell_1_recurrent_kernel-
)assignvariableop_18_gru_1_gru_cell_1_bias
assignvariableop_19_total
assignvariableop_20_count&
"assignvariableop_21_true_positives&
"assignvariableop_22_true_negatives'
#assignvariableop_23_false_positives'
#assignvariableop_24_false_negatives;
7assignvariableop_25_nadam_layer_normalization_1_gamma_m:
6assignvariableop_26_nadam_layer_normalization_1_beta_m.
*assignvariableop_27_nadam_dense_4_kernel_m,
(assignvariableop_28_nadam_dense_4_bias_m.
*assignvariableop_29_nadam_dense_5_kernel_m,
(assignvariableop_30_nadam_dense_5_bias_m.
*assignvariableop_31_nadam_dense_6_kernel_m,
(assignvariableop_32_nadam_dense_6_bias_m.
*assignvariableop_33_nadam_dense_7_kernel_m,
(assignvariableop_34_nadam_dense_7_bias_m7
3assignvariableop_35_nadam_gru_1_gru_cell_1_kernel_mA
=assignvariableop_36_nadam_gru_1_gru_cell_1_recurrent_kernel_m5
1assignvariableop_37_nadam_gru_1_gru_cell_1_bias_m;
7assignvariableop_38_nadam_layer_normalization_1_gamma_v:
6assignvariableop_39_nadam_layer_normalization_1_beta_v.
*assignvariableop_40_nadam_dense_4_kernel_v,
(assignvariableop_41_nadam_dense_4_bias_v.
*assignvariableop_42_nadam_dense_5_kernel_v,
(assignvariableop_43_nadam_dense_5_bias_v.
*assignvariableop_44_nadam_dense_6_kernel_v,
(assignvariableop_45_nadam_dense_6_bias_v.
*assignvariableop_46_nadam_dense_7_kernel_v,
(assignvariableop_47_nadam_dense_7_bias_v7
3assignvariableop_48_nadam_gru_1_gru_cell_1_kernel_vA
=assignvariableop_49_nadam_gru_1_gru_cell_1_recurrent_kernel_v5
1assignvariableop_50_nadam_gru_1_gru_cell_1_bias_v
identity_52??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp,assignvariableop_layer_normalization_1_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp-assignvariableop_1_layer_normalization_1_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_6_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_6_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_7_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_7_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_nadam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_nadam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_nadam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_nadam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_nadam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp(assignvariableop_15_nadam_momentum_cacheIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_gru_1_gru_cell_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp5assignvariableop_17_gru_1_gru_cell_1_recurrent_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_gru_1_gru_cell_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_true_positivesIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_true_negativesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp#assignvariableop_23_false_positivesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_negativesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp7assignvariableop_25_nadam_layer_normalization_1_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp6assignvariableop_26_nadam_layer_normalization_1_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_nadam_dense_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_nadam_dense_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_nadam_dense_5_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_nadam_dense_5_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_nadam_dense_6_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_nadam_dense_6_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_nadam_dense_7_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_nadam_dense_7_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp3assignvariableop_35_nadam_gru_1_gru_cell_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp=assignvariableop_36_nadam_gru_1_gru_cell_1_recurrent_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp1assignvariableop_37_nadam_gru_1_gru_cell_1_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp7assignvariableop_38_nadam_layer_normalization_1_gamma_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp6assignvariableop_39_nadam_layer_normalization_1_beta_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_nadam_dense_4_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_nadam_dense_4_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_nadam_dense_5_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_nadam_dense_5_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_nadam_dense_6_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_nadam_dense_6_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_nadam_dense_7_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_nadam_dense_7_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp3assignvariableop_48_nadam_gru_1_gru_cell_1_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp=assignvariableop_49_nadam_gru_1_gru_cell_1_recurrent_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp1assignvariableop_50_nadam_gru_1_gru_cell_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_509
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51?	
Identity_52IdentityIdentity_51:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_52"#
identity_52Identity_52:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
gru_1_while_cond_223922(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_223922___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_223922___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_223922___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_223922___redundant_placeholder3
gru_1_while_identity
?
gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
gru_1/while/Lesso
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_1/while/Identity"5
gru_1_while_identitygru_1/while/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
޴
?
A__inference_gru_1_layer_call_and_return_conditional_losses_222975

inputs&
"gru_cell_1_readvariableop_resource(
$gru_cell_1_readvariableop_1_resource(
$gru_cell_1_readvariableop_4_resource
identity??gru_cell_1/ReadVariableOp?gru_cell_1/ReadVariableOp_1?gru_cell_1/ReadVariableOp_2?gru_cell_1/ReadVariableOp_3?gru_cell_1/ReadVariableOp_4?gru_cell_1/ReadVariableOp_5?gru_cell_1/ReadVariableOp_6?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????	2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
strided_slice_2v
gru_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_1/ones_like/Shape}
gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/ones_like/Const?
gru_cell_1/ones_likeFill#gru_cell_1/ones_like/Shape:output:0#gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/ones_like?
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_1/ReadVariableOp?
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_1/unstack?
gru_cell_1/ReadVariableOp_1ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_1?
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_1/strided_slice/stack?
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice/stack_1?
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_1/strided_slice/stack_2?
gru_cell_1/strided_sliceStridedSlice#gru_cell_1/ReadVariableOp_1:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0!gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul?
gru_cell_1/ReadVariableOp_2ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_2?
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_1/stack?
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_1/stack_1?
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_1/stack_2?
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_1?
gru_cell_1/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_1?
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_3?
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_1/strided_slice_2/stack?
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_1/strided_slice_2/stack_1?
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_2/stack_2?
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_2?
gru_cell_1/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_2?
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_3/stack?
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_3/stack_1?
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_3/stack_2?
gru_cell_1/strided_slice_3StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_3?
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd?
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_4/stack?
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_4/stack_1?
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_4/stack_2?
gru_cell_1/strided_slice_4StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_4?
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_1?
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_5/stack?
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_1/strided_slice_5/stack_1?
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_5/stack_2?
gru_cell_1/strided_slice_5StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_5?
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_2?
gru_cell_1/mulMulzeros:output:0gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul?
gru_cell_1/mul_1Mulzeros:output:0gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_1?
gru_cell_1/mul_2Mulzeros:output:0gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_2?
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_4?
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_1/strided_slice_6/stack?
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_1/strided_slice_6/stack_1?
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_6/stack_2?
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_6?
gru_cell_1/MatMul_3MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_3?
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_5?
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_7/stack?
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_7/stack_1?
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_7/stack_2?
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_7?
gru_cell_1/MatMul_4MatMulgru_cell_1/mul_1:z:0#gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_4?
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_8/stack?
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_8/stack_1?
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_8/stack_2?
gru_cell_1/strided_slice_8StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_8?
gru_cell_1/BiasAdd_3BiasAddgru_cell_1/MatMul_3:product:0#gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_3?
 gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_9/stack?
"gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_9/stack_1?
"gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_9/stack_2?
gru_cell_1/strided_slice_9StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_9/stack:output:0+gru_cell_1/strided_slice_9/stack_1:output:0+gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_9?
gru_cell_1/BiasAdd_4BiasAddgru_cell_1/MatMul_4:product:0#gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_4?
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/addz
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid?
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_1?
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid_1?
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_6?
!gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_1/strided_slice_10/stack?
#gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_1/strided_slice_10/stack_1?
#gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_1/strided_slice_10/stack_2?
gru_cell_1/strided_slice_10StridedSlice#gru_cell_1/ReadVariableOp_6:value:0*gru_cell_1/strided_slice_10/stack:output:0,gru_cell_1/strided_slice_10/stack_1:output:0,gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_10?
gru_cell_1/MatMul_5MatMulgru_cell_1/mul_2:z:0$gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_5?
!gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_1/strided_slice_11/stack?
#gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_1/strided_slice_11/stack_1?
#gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_1/strided_slice_11/stack_2?
gru_cell_1/strided_slice_11StridedSlicegru_cell_1/unstack:output:1*gru_cell_1/strided_slice_11/stack:output:0,gru_cell_1/strided_slice_11/stack_1:output:0,gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_11?
gru_cell_1/BiasAdd_5BiasAddgru_cell_1/MatMul_5:product:0$gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_5?
gru_cell_1/mul_3Mulgru_cell_1/Sigmoid_1:y:0gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_3?
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_2s
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Tanh?
gru_cell_1/mul_4Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_4i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/sub/x?
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/sub?
gru_cell_1/mul_5Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_5?
gru_cell_1/add_3AddV2gru_cell_1/mul_4:z:0gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_1_resource$gru_cell_1_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_222829*
condR
while_cond_222828*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_223409
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*5J 8? **
f%R#
!__inference__wrapped_model_2216832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
?
?
&__inference_gru_1_layer_call_fn_225425

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_2229752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?%
?
C__inference_model_1_layer_call_and_return_conditional_losses_223188
input_3
input_4
gru_1_222998
gru_1_223000
gru_1_223002 
layer_normalization_1_223058 
layer_normalization_1_223060
dense_4_223085
dense_4_223087
dense_5_223128
dense_5_223130
dense_6_223155
dense_6_223157
dense_7_223182
dense_7_223184
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallinput_3gru_1_222998gru_1_223000gru_1_223002*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_2227042
gru_1/StatefulPartitionedCall?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0layer_normalization_1_223058layer_normalization_1_223060*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_2230472/
-layer_normalization_1/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_4_223085dense_4_223087*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2230742!
dense_4/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*5J 8? *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_2230972
concatenate_1/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_5_223128dense_5_223130*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2231172!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_223155dense_6_223157*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2231442!
dense_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_223182dense_7_223184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2231712!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^gru_1/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
??
?
while_body_222534
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_00
,while_gru_cell_1_readvariableop_1_resource_00
,while_gru_cell_1_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource.
*while_gru_cell_1_readvariableop_1_resource.
*while_gru_cell_1_readvariableop_4_resource??while/gru_cell_1/ReadVariableOp?!while/gru_cell_1/ReadVariableOp_1?!while/gru_cell_1/ReadVariableOp_2?!while/gru_cell_1/ReadVariableOp_3?!while/gru_cell_1/ReadVariableOp_4?!while/gru_cell_1/ReadVariableOp_5?!while/gru_cell_1/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_1/ones_like/Shape?
 while/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_1/ones_like/Const?
while/gru_cell_1/ones_likeFill)while/gru_cell_1/ones_like/Shape:output:0)while/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/ones_like?
while/gru_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell_1/dropout/Const?
while/gru_cell_1/dropout/MulMul#while/gru_cell_1/ones_like:output:0'while/gru_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/dropout/Mul?
while/gru_cell_1/dropout/ShapeShape#while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell_1/dropout/Shape?
5while/gru_cell_1/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell_1/dropout/random_uniform/RandomUniform?
'while/gru_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell_1/dropout/GreaterEqual/y?
%while/gru_cell_1/dropout/GreaterEqualGreaterEqual>while/gru_cell_1/dropout/random_uniform/RandomUniform:output:00while/gru_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell_1/dropout/GreaterEqual?
while/gru_cell_1/dropout/CastCast)while/gru_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell_1/dropout/Cast?
while/gru_cell_1/dropout/Mul_1Mul while/gru_cell_1/dropout/Mul:z:0!while/gru_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_1/dropout/Mul_1?
 while/gru_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_1/dropout_1/Const?
while/gru_cell_1/dropout_1/MulMul#while/gru_cell_1/ones_like:output:0)while/gru_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_1/dropout_1/Mul?
 while/gru_cell_1/dropout_1/ShapeShape#while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_1/dropout_1/Shape?
7while/gru_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??*29
7while/gru_cell_1/dropout_1/random_uniform/RandomUniform?
)while/gru_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_1/dropout_1/GreaterEqual/y?
'while/gru_cell_1/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_1/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_1/dropout_1/GreaterEqual?
while/gru_cell_1/dropout_1/CastCast+while/gru_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_1/dropout_1/Cast?
 while/gru_cell_1/dropout_1/Mul_1Mul"while/gru_cell_1/dropout_1/Mul:z:0#while/gru_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_1/dropout_1/Mul_1?
 while/gru_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_1/dropout_2/Const?
while/gru_cell_1/dropout_2/MulMul#while/gru_cell_1/ones_like:output:0)while/gru_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_1/dropout_2/Mul?
 while/gru_cell_1/dropout_2/ShapeShape#while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_1/dropout_2/Shape?
7while/gru_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???29
7while/gru_cell_1/dropout_2/random_uniform/RandomUniform?
)while/gru_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_1/dropout_2/GreaterEqual/y?
'while/gru_cell_1/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_1/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_1/dropout_2/GreaterEqual?
while/gru_cell_1/dropout_2/CastCast+while/gru_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_1/dropout_2/Cast?
 while/gru_cell_1/dropout_2/Mul_1Mul"while/gru_cell_1/dropout_2/Mul:z:0#while/gru_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_1/dropout_2/Mul_1?
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_1/ReadVariableOp?
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_1/unstack?
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_1?
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_1/strided_slice/stack?
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice/stack_1?
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_1/strided_slice/stack_2?
while/gru_cell_1/strided_sliceStridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_1/strided_slice?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul?
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_2?
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_1/stack?
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_1/stack_1?
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_1/stack_2?
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_1?
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_1?
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_3?
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_1/strided_slice_2/stack?
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_1/strided_slice_2/stack_1?
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_2/stack_2?
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_2?
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_2?
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_3/stack?
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_3/stack_1?
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_3/stack_2?
 while/gru_cell_1/strided_slice_3StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_3?
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd?
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_4/stack?
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_4/stack_1?
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_4/stack_2?
 while/gru_cell_1/strided_slice_4StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_4?
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_1?
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_5/stack?
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_1/strided_slice_5/stack_1?
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_5/stack_2?
 while/gru_cell_1/strided_slice_5StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_1/strided_slice_5?
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_2?
while/gru_cell_1/mulMulwhile_placeholder_2"while/gru_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul?
while/gru_cell_1/mul_1Mulwhile_placeholder_2$while/gru_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_1?
while/gru_cell_1/mul_2Mulwhile_placeholder_2$while/gru_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_2?
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_4?
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_1/strided_slice_6/stack?
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_1/strided_slice_6/stack_1?
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_6/stack_2?
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_6?
while/gru_cell_1/MatMul_3MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_3?
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_5?
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_7/stack?
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_7/stack_1?
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_7/stack_2?
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_7?
while/gru_cell_1/MatMul_4MatMulwhile/gru_cell_1/mul_1:z:0)while/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_4?
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_8/stack?
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_8/stack_1?
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_8/stack_2?
 while/gru_cell_1/strided_slice_8StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_8?
while/gru_cell_1/BiasAdd_3BiasAdd#while/gru_cell_1/MatMul_3:product:0)while/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_3?
&while/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_9/stack?
(while/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_9/stack_1?
(while/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_9/stack_2?
 while/gru_cell_1/strided_slice_9StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_9/stack:output:01while/gru_cell_1/strided_slice_9/stack_1:output:01while/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_9?
while/gru_cell_1/BiasAdd_4BiasAdd#while/gru_cell_1/MatMul_4:product:0)while/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_4?
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add?
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid?
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_1?
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid_1?
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_6?
'while/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_1/strided_slice_10/stack?
)while/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_1/strided_slice_10/stack_1?
)while/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_1/strided_slice_10/stack_2?
!while/gru_cell_1/strided_slice_10StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:00while/gru_cell_1/strided_slice_10/stack:output:02while/gru_cell_1/strided_slice_10/stack_1:output:02while/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_1/strided_slice_10?
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul_2:z:0*while/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_5?
'while/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_1/strided_slice_11/stack?
)while/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_1/strided_slice_11/stack_1?
)while/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_1/strided_slice_11/stack_2?
!while/gru_cell_1/strided_slice_11StridedSlice!while/gru_cell_1/unstack:output:10while/gru_cell_1/strided_slice_11/stack:output:02while/gru_cell_1/strided_slice_11/stack_1:output:02while/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_1/strided_slice_11?
while/gru_cell_1/BiasAdd_5BiasAdd#while/gru_cell_1/MatMul_5:product:0*while/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_5?
while/gru_cell_1/mul_3Mulwhile/gru_cell_1/Sigmoid_1:y:0#while/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_3?
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0while/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_2?
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Tanh?
while/gru_cell_1/mul_4Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_4u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_1/sub/x?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/sub?
while/gru_cell_1/mul_5Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_5?
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_4:z:0while/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_1_readvariableop_1_resource,while_gru_cell_1_readvariableop_1_resource_0"Z
*while_gru_cell_1_readvariableop_4_resource,while_gru_cell_1_readvariableop_4_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?	
?
+__inference_gru_cell_1_layer_call_fn_225799

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_2218352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????	:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
&__inference_gru_1_layer_call_fn_224802
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_2222542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????	
"
_user_specified_name
inputs/0
?
?
6__inference_layer_normalization_1_layer_call_fn_225476

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_2230472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_224644
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_224644___redundant_placeholder04
0while_while_cond_224644___redundant_placeholder14
0while_while_cond_224644___redundant_placeholder24
0while_while_cond_224644___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
!__inference__wrapped_model_221683
input_3
input_44
0model_1_gru_1_gru_cell_1_readvariableop_resource6
2model_1_gru_1_gru_cell_1_readvariableop_1_resource6
2model_1_gru_1_gru_cell_1_readvariableop_4_resource?
;model_1_layer_normalization_1_mul_2_readvariableop_resource=
9model_1_layer_normalization_1_add_readvariableop_resource2
.model_1_dense_4_matmul_readvariableop_resource3
/model_1_dense_4_biasadd_readvariableop_resource2
.model_1_dense_5_matmul_readvariableop_resource3
/model_1_dense_5_biasadd_readvariableop_resource2
.model_1_dense_6_matmul_readvariableop_resource3
/model_1_dense_6_biasadd_readvariableop_resource2
.model_1_dense_7_matmul_readvariableop_resource3
/model_1_dense_7_biasadd_readvariableop_resource
identity??&model_1/dense_4/BiasAdd/ReadVariableOp?%model_1/dense_4/MatMul/ReadVariableOp?&model_1/dense_5/BiasAdd/ReadVariableOp?%model_1/dense_5/MatMul/ReadVariableOp?&model_1/dense_6/BiasAdd/ReadVariableOp?%model_1/dense_6/MatMul/ReadVariableOp?&model_1/dense_7/BiasAdd/ReadVariableOp?%model_1/dense_7/MatMul/ReadVariableOp?'model_1/gru_1/gru_cell_1/ReadVariableOp?)model_1/gru_1/gru_cell_1/ReadVariableOp_1?)model_1/gru_1/gru_cell_1/ReadVariableOp_2?)model_1/gru_1/gru_cell_1/ReadVariableOp_3?)model_1/gru_1/gru_cell_1/ReadVariableOp_4?)model_1/gru_1/gru_cell_1/ReadVariableOp_5?)model_1/gru_1/gru_cell_1/ReadVariableOp_6?model_1/gru_1/while?0model_1/layer_normalization_1/add/ReadVariableOp?2model_1/layer_normalization_1/mul_2/ReadVariableOpa
model_1/gru_1/ShapeShapeinput_3*
T0*
_output_shapes
:2
model_1/gru_1/Shape?
!model_1/gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model_1/gru_1/strided_slice/stack?
#model_1/gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#model_1/gru_1/strided_slice/stack_1?
#model_1/gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model_1/gru_1/strided_slice/stack_2?
model_1/gru_1/strided_sliceStridedSlicemodel_1/gru_1/Shape:output:0*model_1/gru_1/strided_slice/stack:output:0,model_1/gru_1/strided_slice/stack_1:output:0,model_1/gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_1/gru_1/strided_slicey
model_1/gru_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_1/gru_1/zeros/mul/y?
model_1/gru_1/zeros/mulMul$model_1/gru_1/strided_slice:output:0"model_1/gru_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_1/gru_1/zeros/mul{
model_1/gru_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_1/gru_1/zeros/Less/y?
model_1/gru_1/zeros/LessLessmodel_1/gru_1/zeros/mul:z:0#model_1/gru_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_1/gru_1/zeros/Less
model_1/gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
model_1/gru_1/zeros/packed/1?
model_1/gru_1/zeros/packedPack$model_1/gru_1/strided_slice:output:0%model_1/gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_1/gru_1/zeros/packed{
model_1/gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_1/gru_1/zeros/Const?
model_1/gru_1/zerosFill#model_1/gru_1/zeros/packed:output:0"model_1/gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
model_1/gru_1/zeros?
model_1/gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model_1/gru_1/transpose/perm?
model_1/gru_1/transpose	Transposeinput_3%model_1/gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????	2
model_1/gru_1/transposey
model_1/gru_1/Shape_1Shapemodel_1/gru_1/transpose:y:0*
T0*
_output_shapes
:2
model_1/gru_1/Shape_1?
#model_1/gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_1/gru_1/strided_slice_1/stack?
%model_1/gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_1/gru_1/strided_slice_1/stack_1?
%model_1/gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_1/gru_1/strided_slice_1/stack_2?
model_1/gru_1/strided_slice_1StridedSlicemodel_1/gru_1/Shape_1:output:0,model_1/gru_1/strided_slice_1/stack:output:0.model_1/gru_1/strided_slice_1/stack_1:output:0.model_1/gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_1/gru_1/strided_slice_1?
)model_1/gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)model_1/gru_1/TensorArrayV2/element_shape?
model_1/gru_1/TensorArrayV2TensorListReserve2model_1/gru_1/TensorArrayV2/element_shape:output:0&model_1/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_1/gru_1/TensorArrayV2?
Cmodel_1/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2E
Cmodel_1/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
5model_1/gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_1/gru_1/transpose:y:0Lmodel_1/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5model_1/gru_1/TensorArrayUnstack/TensorListFromTensor?
#model_1/gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_1/gru_1/strided_slice_2/stack?
%model_1/gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_1/gru_1/strided_slice_2/stack_1?
%model_1/gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_1/gru_1/strided_slice_2/stack_2?
model_1/gru_1/strided_slice_2StridedSlicemodel_1/gru_1/transpose:y:0,model_1/gru_1/strided_slice_2/stack:output:0.model_1/gru_1/strided_slice_2/stack_1:output:0.model_1/gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
model_1/gru_1/strided_slice_2?
(model_1/gru_1/gru_cell_1/ones_like/ShapeShapemodel_1/gru_1/zeros:output:0*
T0*
_output_shapes
:2*
(model_1/gru_1/gru_cell_1/ones_like/Shape?
(model_1/gru_1/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(model_1/gru_1/gru_cell_1/ones_like/Const?
"model_1/gru_1/gru_cell_1/ones_likeFill1model_1/gru_1/gru_cell_1/ones_like/Shape:output:01model_1/gru_1/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2$
"model_1/gru_1/gru_cell_1/ones_like?
'model_1/gru_1/gru_cell_1/ReadVariableOpReadVariableOp0model_1_gru_1_gru_cell_1_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'model_1/gru_1/gru_cell_1/ReadVariableOp?
 model_1/gru_1/gru_cell_1/unstackUnpack/model_1/gru_1/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 model_1/gru_1/gru_cell_1/unstack?
)model_1/gru_1/gru_cell_1/ReadVariableOp_1ReadVariableOp2model_1_gru_1_gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02+
)model_1/gru_1/gru_cell_1/ReadVariableOp_1?
,model_1/gru_1/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,model_1/gru_1/gru_cell_1/strided_slice/stack?
.model_1/gru_1/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   20
.model_1/gru_1/gru_cell_1/strided_slice/stack_1?
.model_1/gru_1/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_1/gru_1/gru_cell_1/strided_slice/stack_2?
&model_1/gru_1/gru_cell_1/strided_sliceStridedSlice1model_1/gru_1/gru_cell_1/ReadVariableOp_1:value:05model_1/gru_1/gru_cell_1/strided_slice/stack:output:07model_1/gru_1/gru_cell_1/strided_slice/stack_1:output:07model_1/gru_1/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&model_1/gru_1/gru_cell_1/strided_slice?
model_1/gru_1/gru_cell_1/MatMulMatMul&model_1/gru_1/strided_slice_2:output:0/model_1/gru_1/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2!
model_1/gru_1/gru_cell_1/MatMul?
)model_1/gru_1/gru_cell_1/ReadVariableOp_2ReadVariableOp2model_1_gru_1_gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02+
)model_1/gru_1/gru_cell_1/ReadVariableOp_2?
.model_1/gru_1/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   20
.model_1/gru_1/gru_cell_1/strided_slice_1/stack?
0model_1/gru_1/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  22
0model_1/gru_1/gru_cell_1/strided_slice_1/stack_1?
0model_1/gru_1/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model_1/gru_1/gru_cell_1/strided_slice_1/stack_2?
(model_1/gru_1/gru_cell_1/strided_slice_1StridedSlice1model_1/gru_1/gru_cell_1/ReadVariableOp_2:value:07model_1/gru_1/gru_cell_1/strided_slice_1/stack:output:09model_1/gru_1/gru_cell_1/strided_slice_1/stack_1:output:09model_1/gru_1/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2*
(model_1/gru_1/gru_cell_1/strided_slice_1?
!model_1/gru_1/gru_cell_1/MatMul_1MatMul&model_1/gru_1/strided_slice_2:output:01model_1/gru_1/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2#
!model_1/gru_1/gru_cell_1/MatMul_1?
)model_1/gru_1/gru_cell_1/ReadVariableOp_3ReadVariableOp2model_1_gru_1_gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02+
)model_1/gru_1/gru_cell_1/ReadVariableOp_3?
.model_1/gru_1/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  20
.model_1/gru_1/gru_cell_1/strided_slice_2/stack?
0model_1/gru_1/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0model_1/gru_1/gru_cell_1/strided_slice_2/stack_1?
0model_1/gru_1/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model_1/gru_1/gru_cell_1/strided_slice_2/stack_2?
(model_1/gru_1/gru_cell_1/strided_slice_2StridedSlice1model_1/gru_1/gru_cell_1/ReadVariableOp_3:value:07model_1/gru_1/gru_cell_1/strided_slice_2/stack:output:09model_1/gru_1/gru_cell_1/strided_slice_2/stack_1:output:09model_1/gru_1/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2*
(model_1/gru_1/gru_cell_1/strided_slice_2?
!model_1/gru_1/gru_cell_1/MatMul_2MatMul&model_1/gru_1/strided_slice_2:output:01model_1/gru_1/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2#
!model_1/gru_1/gru_cell_1/MatMul_2?
.model_1/gru_1/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/gru_1/gru_cell_1/strided_slice_3/stack?
0model_1/gru_1/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model_1/gru_1/gru_cell_1/strided_slice_3/stack_1?
0model_1/gru_1/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/gru_1/gru_cell_1/strided_slice_3/stack_2?
(model_1/gru_1/gru_cell_1/strided_slice_3StridedSlice)model_1/gru_1/gru_cell_1/unstack:output:07model_1/gru_1/gru_cell_1/strided_slice_3/stack:output:09model_1/gru_1/gru_cell_1/strided_slice_3/stack_1:output:09model_1/gru_1/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2*
(model_1/gru_1/gru_cell_1/strided_slice_3?
 model_1/gru_1/gru_cell_1/BiasAddBiasAdd)model_1/gru_1/gru_cell_1/MatMul:product:01model_1/gru_1/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2"
 model_1/gru_1/gru_cell_1/BiasAdd?
.model_1/gru_1/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?20
.model_1/gru_1/gru_cell_1/strided_slice_4/stack?
0model_1/gru_1/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model_1/gru_1/gru_cell_1/strided_slice_4/stack_1?
0model_1/gru_1/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/gru_1/gru_cell_1/strided_slice_4/stack_2?
(model_1/gru_1/gru_cell_1/strided_slice_4StridedSlice)model_1/gru_1/gru_cell_1/unstack:output:07model_1/gru_1/gru_cell_1/strided_slice_4/stack:output:09model_1/gru_1/gru_cell_1/strided_slice_4/stack_1:output:09model_1/gru_1/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2*
(model_1/gru_1/gru_cell_1/strided_slice_4?
"model_1/gru_1/gru_cell_1/BiasAdd_1BiasAdd+model_1/gru_1/gru_cell_1/MatMul_1:product:01model_1/gru_1/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2$
"model_1/gru_1/gru_cell_1/BiasAdd_1?
.model_1/gru_1/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?20
.model_1/gru_1/gru_cell_1/strided_slice_5/stack?
0model_1/gru_1/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/gru_1/gru_cell_1/strided_slice_5/stack_1?
0model_1/gru_1/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/gru_1/gru_cell_1/strided_slice_5/stack_2?
(model_1/gru_1/gru_cell_1/strided_slice_5StridedSlice)model_1/gru_1/gru_cell_1/unstack:output:07model_1/gru_1/gru_cell_1/strided_slice_5/stack:output:09model_1/gru_1/gru_cell_1/strided_slice_5/stack_1:output:09model_1/gru_1/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2*
(model_1/gru_1/gru_cell_1/strided_slice_5?
"model_1/gru_1/gru_cell_1/BiasAdd_2BiasAdd+model_1/gru_1/gru_cell_1/MatMul_2:product:01model_1/gru_1/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2$
"model_1/gru_1/gru_cell_1/BiasAdd_2?
model_1/gru_1/gru_cell_1/mulMulmodel_1/gru_1/zeros:output:0+model_1/gru_1/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
model_1/gru_1/gru_cell_1/mul?
model_1/gru_1/gru_cell_1/mul_1Mulmodel_1/gru_1/zeros:output:0+model_1/gru_1/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2 
model_1/gru_1/gru_cell_1/mul_1?
model_1/gru_1/gru_cell_1/mul_2Mulmodel_1/gru_1/zeros:output:0+model_1/gru_1/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2 
model_1/gru_1/gru_cell_1/mul_2?
)model_1/gru_1/gru_cell_1/ReadVariableOp_4ReadVariableOp2model_1_gru_1_gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02+
)model_1/gru_1/gru_cell_1/ReadVariableOp_4?
.model_1/gru_1/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.model_1/gru_1/gru_cell_1/strided_slice_6/stack?
0model_1/gru_1/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0model_1/gru_1/gru_cell_1/strided_slice_6/stack_1?
0model_1/gru_1/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model_1/gru_1/gru_cell_1/strided_slice_6/stack_2?
(model_1/gru_1/gru_cell_1/strided_slice_6StridedSlice1model_1/gru_1/gru_cell_1/ReadVariableOp_4:value:07model_1/gru_1/gru_cell_1/strided_slice_6/stack:output:09model_1/gru_1/gru_cell_1/strided_slice_6/stack_1:output:09model_1/gru_1/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(model_1/gru_1/gru_cell_1/strided_slice_6?
!model_1/gru_1/gru_cell_1/MatMul_3MatMul model_1/gru_1/gru_cell_1/mul:z:01model_1/gru_1/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2#
!model_1/gru_1/gru_cell_1/MatMul_3?
)model_1/gru_1/gru_cell_1/ReadVariableOp_5ReadVariableOp2model_1_gru_1_gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02+
)model_1/gru_1/gru_cell_1/ReadVariableOp_5?
.model_1/gru_1/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   20
.model_1/gru_1/gru_cell_1/strided_slice_7/stack?
0model_1/gru_1/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  22
0model_1/gru_1/gru_cell_1/strided_slice_7/stack_1?
0model_1/gru_1/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model_1/gru_1/gru_cell_1/strided_slice_7/stack_2?
(model_1/gru_1/gru_cell_1/strided_slice_7StridedSlice1model_1/gru_1/gru_cell_1/ReadVariableOp_5:value:07model_1/gru_1/gru_cell_1/strided_slice_7/stack:output:09model_1/gru_1/gru_cell_1/strided_slice_7/stack_1:output:09model_1/gru_1/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(model_1/gru_1/gru_cell_1/strided_slice_7?
!model_1/gru_1/gru_cell_1/MatMul_4MatMul"model_1/gru_1/gru_cell_1/mul_1:z:01model_1/gru_1/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2#
!model_1/gru_1/gru_cell_1/MatMul_4?
.model_1/gru_1/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/gru_1/gru_cell_1/strided_slice_8/stack?
0model_1/gru_1/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model_1/gru_1/gru_cell_1/strided_slice_8/stack_1?
0model_1/gru_1/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/gru_1/gru_cell_1/strided_slice_8/stack_2?
(model_1/gru_1/gru_cell_1/strided_slice_8StridedSlice)model_1/gru_1/gru_cell_1/unstack:output:17model_1/gru_1/gru_cell_1/strided_slice_8/stack:output:09model_1/gru_1/gru_cell_1/strided_slice_8/stack_1:output:09model_1/gru_1/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2*
(model_1/gru_1/gru_cell_1/strided_slice_8?
"model_1/gru_1/gru_cell_1/BiasAdd_3BiasAdd+model_1/gru_1/gru_cell_1/MatMul_3:product:01model_1/gru_1/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2$
"model_1/gru_1/gru_cell_1/BiasAdd_3?
.model_1/gru_1/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?20
.model_1/gru_1/gru_cell_1/strided_slice_9/stack?
0model_1/gru_1/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model_1/gru_1/gru_cell_1/strided_slice_9/stack_1?
0model_1/gru_1/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/gru_1/gru_cell_1/strided_slice_9/stack_2?
(model_1/gru_1/gru_cell_1/strided_slice_9StridedSlice)model_1/gru_1/gru_cell_1/unstack:output:17model_1/gru_1/gru_cell_1/strided_slice_9/stack:output:09model_1/gru_1/gru_cell_1/strided_slice_9/stack_1:output:09model_1/gru_1/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2*
(model_1/gru_1/gru_cell_1/strided_slice_9?
"model_1/gru_1/gru_cell_1/BiasAdd_4BiasAdd+model_1/gru_1/gru_cell_1/MatMul_4:product:01model_1/gru_1/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2$
"model_1/gru_1/gru_cell_1/BiasAdd_4?
model_1/gru_1/gru_cell_1/addAddV2)model_1/gru_1/gru_cell_1/BiasAdd:output:0+model_1/gru_1/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
model_1/gru_1/gru_cell_1/add?
 model_1/gru_1/gru_cell_1/SigmoidSigmoid model_1/gru_1/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2"
 model_1/gru_1/gru_cell_1/Sigmoid?
model_1/gru_1/gru_cell_1/add_1AddV2+model_1/gru_1/gru_cell_1/BiasAdd_1:output:0+model_1/gru_1/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2 
model_1/gru_1/gru_cell_1/add_1?
"model_1/gru_1/gru_cell_1/Sigmoid_1Sigmoid"model_1/gru_1/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"model_1/gru_1/gru_cell_1/Sigmoid_1?
)model_1/gru_1/gru_cell_1/ReadVariableOp_6ReadVariableOp2model_1_gru_1_gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02+
)model_1/gru_1/gru_cell_1/ReadVariableOp_6?
/model_1/gru_1/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  21
/model_1/gru_1/gru_cell_1/strided_slice_10/stack?
1model_1/gru_1/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1model_1/gru_1/gru_cell_1/strided_slice_10/stack_1?
1model_1/gru_1/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1model_1/gru_1/gru_cell_1/strided_slice_10/stack_2?
)model_1/gru_1/gru_cell_1/strided_slice_10StridedSlice1model_1/gru_1/gru_cell_1/ReadVariableOp_6:value:08model_1/gru_1/gru_cell_1/strided_slice_10/stack:output:0:model_1/gru_1/gru_cell_1/strided_slice_10/stack_1:output:0:model_1/gru_1/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2+
)model_1/gru_1/gru_cell_1/strided_slice_10?
!model_1/gru_1/gru_cell_1/MatMul_5MatMul"model_1/gru_1/gru_cell_1/mul_2:z:02model_1/gru_1/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2#
!model_1/gru_1/gru_cell_1/MatMul_5?
/model_1/gru_1/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?21
/model_1/gru_1/gru_cell_1/strided_slice_11/stack?
1model_1/gru_1/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1model_1/gru_1/gru_cell_1/strided_slice_11/stack_1?
1model_1/gru_1/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_1/gru_1/gru_cell_1/strided_slice_11/stack_2?
)model_1/gru_1/gru_cell_1/strided_slice_11StridedSlice)model_1/gru_1/gru_cell_1/unstack:output:18model_1/gru_1/gru_cell_1/strided_slice_11/stack:output:0:model_1/gru_1/gru_cell_1/strided_slice_11/stack_1:output:0:model_1/gru_1/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2+
)model_1/gru_1/gru_cell_1/strided_slice_11?
"model_1/gru_1/gru_cell_1/BiasAdd_5BiasAdd+model_1/gru_1/gru_cell_1/MatMul_5:product:02model_1/gru_1/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2$
"model_1/gru_1/gru_cell_1/BiasAdd_5?
model_1/gru_1/gru_cell_1/mul_3Mul&model_1/gru_1/gru_cell_1/Sigmoid_1:y:0+model_1/gru_1/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2 
model_1/gru_1/gru_cell_1/mul_3?
model_1/gru_1/gru_cell_1/add_2AddV2+model_1/gru_1/gru_cell_1/BiasAdd_2:output:0"model_1/gru_1/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2 
model_1/gru_1/gru_cell_1/add_2?
model_1/gru_1/gru_cell_1/TanhTanh"model_1/gru_1/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
model_1/gru_1/gru_cell_1/Tanh?
model_1/gru_1/gru_cell_1/mul_4Mul$model_1/gru_1/gru_cell_1/Sigmoid:y:0model_1/gru_1/zeros:output:0*
T0*(
_output_shapes
:??????????2 
model_1/gru_1/gru_cell_1/mul_4?
model_1/gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
model_1/gru_1/gru_cell_1/sub/x?
model_1/gru_1/gru_cell_1/subSub'model_1/gru_1/gru_cell_1/sub/x:output:0$model_1/gru_1/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
model_1/gru_1/gru_cell_1/sub?
model_1/gru_1/gru_cell_1/mul_5Mul model_1/gru_1/gru_cell_1/sub:z:0!model_1/gru_1/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2 
model_1/gru_1/gru_cell_1/mul_5?
model_1/gru_1/gru_cell_1/add_3AddV2"model_1/gru_1/gru_cell_1/mul_4:z:0"model_1/gru_1/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2 
model_1/gru_1/gru_cell_1/add_3?
+model_1/gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2-
+model_1/gru_1/TensorArrayV2_1/element_shape?
model_1/gru_1/TensorArrayV2_1TensorListReserve4model_1/gru_1/TensorArrayV2_1/element_shape:output:0&model_1/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_1/gru_1/TensorArrayV2_1j
model_1/gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_1/gru_1/time?
&model_1/gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&model_1/gru_1/while/maximum_iterations?
 model_1/gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 model_1/gru_1/while/loop_counter?
model_1/gru_1/whileWhile)model_1/gru_1/while/loop_counter:output:0/model_1/gru_1/while/maximum_iterations:output:0model_1/gru_1/time:output:0&model_1/gru_1/TensorArrayV2_1:handle:0model_1/gru_1/zeros:output:0&model_1/gru_1/strided_slice_1:output:0Emodel_1/gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00model_1_gru_1_gru_cell_1_readvariableop_resource2model_1_gru_1_gru_cell_1_readvariableop_1_resource2model_1_gru_1_gru_cell_1_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*+
body#R!
model_1_gru_1_while_body_221469*+
cond#R!
model_1_gru_1_while_cond_221468*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
model_1/gru_1/while?
>model_1/gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>model_1/gru_1/TensorArrayV2Stack/TensorListStack/element_shape?
0model_1/gru_1/TensorArrayV2Stack/TensorListStackTensorListStackmodel_1/gru_1/while:output:3Gmodel_1/gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype022
0model_1/gru_1/TensorArrayV2Stack/TensorListStack?
#model_1/gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#model_1/gru_1/strided_slice_3/stack?
%model_1/gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_1/gru_1/strided_slice_3/stack_1?
%model_1/gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_1/gru_1/strided_slice_3/stack_2?
model_1/gru_1/strided_slice_3StridedSlice9model_1/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0,model_1/gru_1/strided_slice_3/stack:output:0.model_1/gru_1/strided_slice_3/stack_1:output:0.model_1/gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
model_1/gru_1/strided_slice_3?
model_1/gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model_1/gru_1/transpose_1/perm?
model_1/gru_1/transpose_1	Transpose9model_1/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0'model_1/gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
model_1/gru_1/transpose_1?
model_1/gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_1/gru_1/runtime?
#model_1/layer_normalization_1/ShapeShape&model_1/gru_1/strided_slice_3:output:0*
T0*
_output_shapes
:2%
#model_1/layer_normalization_1/Shape?
1model_1/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_1/layer_normalization_1/strided_slice/stack?
3model_1/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_1/layer_normalization_1/strided_slice/stack_1?
3model_1/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_1/layer_normalization_1/strided_slice/stack_2?
+model_1/layer_normalization_1/strided_sliceStridedSlice,model_1/layer_normalization_1/Shape:output:0:model_1/layer_normalization_1/strided_slice/stack:output:0<model_1/layer_normalization_1/strided_slice/stack_1:output:0<model_1/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_1/layer_normalization_1/strided_slice?
#model_1/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_1/layer_normalization_1/mul/x?
!model_1/layer_normalization_1/mulMul,model_1/layer_normalization_1/mul/x:output:04model_1/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 2#
!model_1/layer_normalization_1/mul?
3model_1/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3model_1/layer_normalization_1/strided_slice_1/stack?
5model_1/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5model_1/layer_normalization_1/strided_slice_1/stack_1?
5model_1/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5model_1/layer_normalization_1/strided_slice_1/stack_2?
-model_1/layer_normalization_1/strided_slice_1StridedSlice,model_1/layer_normalization_1/Shape:output:0<model_1/layer_normalization_1/strided_slice_1/stack:output:0>model_1/layer_normalization_1/strided_slice_1/stack_1:output:0>model_1/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-model_1/layer_normalization_1/strided_slice_1?
%model_1/layer_normalization_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_1/layer_normalization_1/mul_1/x?
#model_1/layer_normalization_1/mul_1Mul.model_1/layer_normalization_1/mul_1/x:output:06model_1/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2%
#model_1/layer_normalization_1/mul_1?
-model_1/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_1/layer_normalization_1/Reshape/shape/0?
-model_1/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_1/layer_normalization_1/Reshape/shape/3?
+model_1/layer_normalization_1/Reshape/shapePack6model_1/layer_normalization_1/Reshape/shape/0:output:0%model_1/layer_normalization_1/mul:z:0'model_1/layer_normalization_1/mul_1:z:06model_1/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+model_1/layer_normalization_1/Reshape/shape?
%model_1/layer_normalization_1/ReshapeReshape&model_1/gru_1/strided_slice_3:output:04model_1/layer_normalization_1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%model_1/layer_normalization_1/Reshape?
#model_1/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#model_1/layer_normalization_1/Const?
'model_1/layer_normalization_1/Fill/dimsPack%model_1/layer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:2)
'model_1/layer_normalization_1/Fill/dims?
"model_1/layer_normalization_1/FillFill0model_1/layer_normalization_1/Fill/dims:output:0,model_1/layer_normalization_1/Const:output:0*
T0*#
_output_shapes
:?????????2$
"model_1/layer_normalization_1/Fill?
%model_1/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2'
%model_1/layer_normalization_1/Const_1?
)model_1/layer_normalization_1/Fill_1/dimsPack%model_1/layer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:2+
)model_1/layer_normalization_1/Fill_1/dims?
$model_1/layer_normalization_1/Fill_1Fill2model_1/layer_normalization_1/Fill_1/dims:output:0.model_1/layer_normalization_1/Const_1:output:0*
T0*#
_output_shapes
:?????????2&
$model_1/layer_normalization_1/Fill_1?
%model_1/layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2'
%model_1/layer_normalization_1/Const_2?
%model_1/layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2'
%model_1/layer_normalization_1/Const_3?
.model_1/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3.model_1/layer_normalization_1/Reshape:output:0+model_1/layer_normalization_1/Fill:output:0-model_1/layer_normalization_1/Fill_1:output:0.model_1/layer_normalization_1/Const_2:output:0.model_1/layer_normalization_1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:20
.model_1/layer_normalization_1/FusedBatchNormV3?
'model_1/layer_normalization_1/Reshape_1Reshape2model_1/layer_normalization_1/FusedBatchNormV3:y:0,model_1/layer_normalization_1/Shape:output:0*
T0*(
_output_shapes
:??????????2)
'model_1/layer_normalization_1/Reshape_1?
2model_1/layer_normalization_1/mul_2/ReadVariableOpReadVariableOp;model_1_layer_normalization_1_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_1/layer_normalization_1/mul_2/ReadVariableOp?
#model_1/layer_normalization_1/mul_2Mul0model_1/layer_normalization_1/Reshape_1:output:0:model_1/layer_normalization_1/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#model_1/layer_normalization_1/mul_2?
0model_1/layer_normalization_1/add/ReadVariableOpReadVariableOp9model_1_layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_1/layer_normalization_1/add/ReadVariableOp?
!model_1/layer_normalization_1/addAddV2'model_1/layer_normalization_1/mul_2:z:08model_1/layer_normalization_1/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!model_1/layer_normalization_1/add?
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%model_1/dense_4/MatMul/ReadVariableOp?
model_1/dense_4/MatMulMatMulinput_4-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_4/MatMul?
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_1/dense_4/BiasAdd/ReadVariableOp?
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_4/BiasAdd?
model_1/dense_4/SeluSelu model_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_1/dense_4/Selu?
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_1/concat/axis?
model_1/concatenate_1/concatConcatV2%model_1/layer_normalization_1/add:z:0"model_1/dense_4/Selu:activations:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model_1/concatenate_1/concat?
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%model_1/dense_5/MatMul/ReadVariableOp?
model_1/dense_5/MatMulMatMul%model_1/concatenate_1/concat:output:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_5/MatMul?
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_1/dense_5/BiasAdd/ReadVariableOp?
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_5/BiasAdd?
model_1/dense_5/SeluSelu model_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_1/dense_5/Selu?
%model_1/dense_6/MatMul/ReadVariableOpReadVariableOp.model_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%model_1/dense_6/MatMul/ReadVariableOp?
model_1/dense_6/MatMulMatMul"model_1/dense_5/Selu:activations:0-model_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_6/MatMul?
&model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_1/dense_6/BiasAdd/ReadVariableOp?
model_1/dense_6/BiasAddBiasAdd model_1/dense_6/MatMul:product:0.model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_6/BiasAdd?
model_1/dense_6/SeluSelu model_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_1/dense_6/Selu?
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%model_1/dense_7/MatMul/ReadVariableOp?
model_1/dense_7/MatMulMatMul"model_1/dense_6/Selu:activations:0-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_7/MatMul?
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_7/BiasAdd/ReadVariableOp?
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_7/BiasAdd?
model_1/dense_7/SigmoidSigmoid model_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_7/Sigmoid?
IdentityIdentitymodel_1/dense_7/Sigmoid:y:0'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp'^model_1/dense_6/BiasAdd/ReadVariableOp&^model_1/dense_6/MatMul/ReadVariableOp'^model_1/dense_7/BiasAdd/ReadVariableOp&^model_1/dense_7/MatMul/ReadVariableOp(^model_1/gru_1/gru_cell_1/ReadVariableOp*^model_1/gru_1/gru_cell_1/ReadVariableOp_1*^model_1/gru_1/gru_cell_1/ReadVariableOp_2*^model_1/gru_1/gru_cell_1/ReadVariableOp_3*^model_1/gru_1/gru_cell_1/ReadVariableOp_4*^model_1/gru_1/gru_cell_1/ReadVariableOp_5*^model_1/gru_1/gru_cell_1/ReadVariableOp_6^model_1/gru_1/while1^model_1/layer_normalization_1/add/ReadVariableOp3^model_1/layer_normalization_1/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2P
&model_1/dense_6/BiasAdd/ReadVariableOp&model_1/dense_6/BiasAdd/ReadVariableOp2N
%model_1/dense_6/MatMul/ReadVariableOp%model_1/dense_6/MatMul/ReadVariableOp2P
&model_1/dense_7/BiasAdd/ReadVariableOp&model_1/dense_7/BiasAdd/ReadVariableOp2N
%model_1/dense_7/MatMul/ReadVariableOp%model_1/dense_7/MatMul/ReadVariableOp2R
'model_1/gru_1/gru_cell_1/ReadVariableOp'model_1/gru_1/gru_cell_1/ReadVariableOp2V
)model_1/gru_1/gru_cell_1/ReadVariableOp_1)model_1/gru_1/gru_cell_1/ReadVariableOp_12V
)model_1/gru_1/gru_cell_1/ReadVariableOp_2)model_1/gru_1/gru_cell_1/ReadVariableOp_22V
)model_1/gru_1/gru_cell_1/ReadVariableOp_3)model_1/gru_1/gru_cell_1/ReadVariableOp_32V
)model_1/gru_1/gru_cell_1/ReadVariableOp_4)model_1/gru_1/gru_cell_1/ReadVariableOp_42V
)model_1/gru_1/gru_cell_1/ReadVariableOp_5)model_1/gru_1/gru_cell_1/ReadVariableOp_52V
)model_1/gru_1/gru_cell_1/ReadVariableOp_6)model_1/gru_1/gru_cell_1/ReadVariableOp_62*
model_1/gru_1/whilemodel_1/gru_1/while2d
0model_1/layer_normalization_1/add/ReadVariableOp0model_1/layer_normalization_1/add/ReadVariableOp2h
2model_1/layer_normalization_1/mul_2/ReadVariableOp2model_1/layer_normalization_1/mul_2/ReadVariableOp:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
??
?
A__inference_gru_1_layer_call_and_return_conditional_losses_222704

inputs&
"gru_cell_1_readvariableop_resource(
$gru_cell_1_readvariableop_1_resource(
$gru_cell_1_readvariableop_4_resource
identity??gru_cell_1/ReadVariableOp?gru_cell_1/ReadVariableOp_1?gru_cell_1/ReadVariableOp_2?gru_cell_1/ReadVariableOp_3?gru_cell_1/ReadVariableOp_4?gru_cell_1/ReadVariableOp_5?gru_cell_1/ReadVariableOp_6?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????	2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
strided_slice_2v
gru_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_1/ones_like/Shape}
gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/ones_like/Const?
gru_cell_1/ones_likeFill#gru_cell_1/ones_like/Shape:output:0#gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/ones_likey
gru_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_1/dropout/Const?
gru_cell_1/dropout/MulMulgru_cell_1/ones_like:output:0!gru_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout/Mul?
gru_cell_1/dropout/ShapeShapegru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_1/dropout/Shape?
/gru_cell_1/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??\21
/gru_cell_1/dropout/random_uniform/RandomUniform?
!gru_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell_1/dropout/GreaterEqual/y?
gru_cell_1/dropout/GreaterEqualGreaterEqual8gru_cell_1/dropout/random_uniform/RandomUniform:output:0*gru_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell_1/dropout/GreaterEqual?
gru_cell_1/dropout/CastCast#gru_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_1/dropout/Cast?
gru_cell_1/dropout/Mul_1Mulgru_cell_1/dropout/Mul:z:0gru_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout/Mul_1}
gru_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_1/dropout_1/Const?
gru_cell_1/dropout_1/MulMulgru_cell_1/ones_like:output:0#gru_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_1/Mul?
gru_cell_1/dropout_1/ShapeShapegru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_1/dropout_1/Shape?
1gru_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???23
1gru_cell_1/dropout_1/random_uniform/RandomUniform?
#gru_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_1/dropout_1/GreaterEqual/y?
!gru_cell_1/dropout_1/GreaterEqualGreaterEqual:gru_cell_1/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_1/dropout_1/GreaterEqual?
gru_cell_1/dropout_1/CastCast%gru_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_1/dropout_1/Cast?
gru_cell_1/dropout_1/Mul_1Mulgru_cell_1/dropout_1/Mul:z:0gru_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_1/Mul_1}
gru_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_1/dropout_2/Const?
gru_cell_1/dropout_2/MulMulgru_cell_1/ones_like:output:0#gru_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_2/Mul?
gru_cell_1/dropout_2/ShapeShapegru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_1/dropout_2/Shape?
1gru_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2Ð?23
1gru_cell_1/dropout_2/random_uniform/RandomUniform?
#gru_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_1/dropout_2/GreaterEqual/y?
!gru_cell_1/dropout_2/GreaterEqualGreaterEqual:gru_cell_1/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_1/dropout_2/GreaterEqual?
gru_cell_1/dropout_2/CastCast%gru_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_1/dropout_2/Cast?
gru_cell_1/dropout_2/Mul_1Mulgru_cell_1/dropout_2/Mul:z:0gru_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/dropout_2/Mul_1?
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_1/ReadVariableOp?
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_1/unstack?
gru_cell_1/ReadVariableOp_1ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_1?
gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_1/strided_slice/stack?
 gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice/stack_1?
 gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_1/strided_slice/stack_2?
gru_cell_1/strided_sliceStridedSlice#gru_cell_1/ReadVariableOp_1:value:0'gru_cell_1/strided_slice/stack:output:0)gru_cell_1/strided_slice/stack_1:output:0)gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0!gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul?
gru_cell_1/ReadVariableOp_2ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_2?
 gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_1/stack?
"gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_1/stack_1?
"gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_1/stack_2?
gru_cell_1/strided_slice_1StridedSlice#gru_cell_1/ReadVariableOp_2:value:0)gru_cell_1/strided_slice_1/stack:output:0+gru_cell_1/strided_slice_1/stack_1:output:0+gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_1?
gru_cell_1/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_1?
gru_cell_1/ReadVariableOp_3ReadVariableOp$gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_1/ReadVariableOp_3?
 gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_1/strided_slice_2/stack?
"gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_1/strided_slice_2/stack_1?
"gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_2/stack_2?
gru_cell_1/strided_slice_2StridedSlice#gru_cell_1/ReadVariableOp_3:value:0)gru_cell_1/strided_slice_2/stack:output:0+gru_cell_1/strided_slice_2/stack_1:output:0+gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_1/strided_slice_2?
gru_cell_1/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_2?
 gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_3/stack?
"gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_3/stack_1?
"gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_3/stack_2?
gru_cell_1/strided_slice_3StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_3/stack:output:0+gru_cell_1/strided_slice_3/stack_1:output:0+gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_3?
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0#gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd?
 gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_4/stack?
"gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_4/stack_1?
"gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_4/stack_2?
gru_cell_1/strided_slice_4StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_4/stack:output:0+gru_cell_1/strided_slice_4/stack_1:output:0+gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_4?
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0#gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_1?
 gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_5/stack?
"gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_1/strided_slice_5/stack_1?
"gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_5/stack_2?
gru_cell_1/strided_slice_5StridedSlicegru_cell_1/unstack:output:0)gru_cell_1/strided_slice_5/stack:output:0+gru_cell_1/strided_slice_5/stack_1:output:0+gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_5?
gru_cell_1/BiasAdd_2BiasAddgru_cell_1/MatMul_2:product:0#gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_2?
gru_cell_1/mulMulzeros:output:0gru_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul?
gru_cell_1/mul_1Mulzeros:output:0gru_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_1?
gru_cell_1/mul_2Mulzeros:output:0gru_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_2?
gru_cell_1/ReadVariableOp_4ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_4?
 gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_1/strided_slice_6/stack?
"gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_1/strided_slice_6/stack_1?
"gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_6/stack_2?
gru_cell_1/strided_slice_6StridedSlice#gru_cell_1/ReadVariableOp_4:value:0)gru_cell_1/strided_slice_6/stack:output:0+gru_cell_1/strided_slice_6/stack_1:output:0+gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_6?
gru_cell_1/MatMul_3MatMulgru_cell_1/mul:z:0#gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_3?
gru_cell_1/ReadVariableOp_5ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_5?
 gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_1/strided_slice_7/stack?
"gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_1/strided_slice_7/stack_1?
"gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_1/strided_slice_7/stack_2?
gru_cell_1/strided_slice_7StridedSlice#gru_cell_1/ReadVariableOp_5:value:0)gru_cell_1/strided_slice_7/stack:output:0+gru_cell_1/strided_slice_7/stack_1:output:0+gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_7?
gru_cell_1/MatMul_4MatMulgru_cell_1/mul_1:z:0#gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_4?
 gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_1/strided_slice_8/stack?
"gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_8/stack_1?
"gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_8/stack_2?
gru_cell_1/strided_slice_8StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_8/stack:output:0+gru_cell_1/strided_slice_8/stack_1:output:0+gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_1/strided_slice_8?
gru_cell_1/BiasAdd_3BiasAddgru_cell_1/MatMul_3:product:0#gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_3?
 gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_1/strided_slice_9/stack?
"gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_1/strided_slice_9/stack_1?
"gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_1/strided_slice_9/stack_2?
gru_cell_1/strided_slice_9StridedSlicegru_cell_1/unstack:output:1)gru_cell_1/strided_slice_9/stack:output:0+gru_cell_1/strided_slice_9/stack_1:output:0+gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_1/strided_slice_9?
gru_cell_1/BiasAdd_4BiasAddgru_cell_1/MatMul_4:product:0#gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_4?
gru_cell_1/addAddV2gru_cell_1/BiasAdd:output:0gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/addz
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid?
gru_cell_1/add_1AddV2gru_cell_1/BiasAdd_1:output:0gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_1?
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Sigmoid_1?
gru_cell_1/ReadVariableOp_6ReadVariableOp$gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_1/ReadVariableOp_6?
!gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_1/strided_slice_10/stack?
#gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_1/strided_slice_10/stack_1?
#gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_1/strided_slice_10/stack_2?
gru_cell_1/strided_slice_10StridedSlice#gru_cell_1/ReadVariableOp_6:value:0*gru_cell_1/strided_slice_10/stack:output:0,gru_cell_1/strided_slice_10/stack_1:output:0,gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_1/strided_slice_10?
gru_cell_1/MatMul_5MatMulgru_cell_1/mul_2:z:0$gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/MatMul_5?
!gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_1/strided_slice_11/stack?
#gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_1/strided_slice_11/stack_1?
#gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_1/strided_slice_11/stack_2?
gru_cell_1/strided_slice_11StridedSlicegru_cell_1/unstack:output:1*gru_cell_1/strided_slice_11/stack:output:0,gru_cell_1/strided_slice_11/stack_1:output:0,gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_1/strided_slice_11?
gru_cell_1/BiasAdd_5BiasAddgru_cell_1/MatMul_5:product:0$gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/BiasAdd_5?
gru_cell_1/mul_3Mulgru_cell_1/Sigmoid_1:y:0gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_3?
gru_cell_1/add_2AddV2gru_cell_1/BiasAdd_2:output:0gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_2s
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/Tanh?
gru_cell_1/mul_4Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_4i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_1/sub/x?
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/sub?
gru_cell_1/mul_5Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/mul_5?
gru_cell_1/add_3AddV2gru_cell_1/mul_4:z:0gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_1/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource$gru_cell_1_readvariableop_1_resource$gru_cell_1_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_222534*
condR
while_cond_222533*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell_1/ReadVariableOp^gru_cell_1/ReadVariableOp_1^gru_cell_1/ReadVariableOp_2^gru_cell_1/ReadVariableOp_3^gru_cell_1/ReadVariableOp_4^gru_cell_1/ReadVariableOp_5^gru_cell_1/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2:
gru_cell_1/ReadVariableOp_1gru_cell_1/ReadVariableOp_12:
gru_cell_1/ReadVariableOp_2gru_cell_1/ReadVariableOp_22:
gru_cell_1/ReadVariableOp_3gru_cell_1/ReadVariableOp_32:
gru_cell_1/ReadVariableOp_4gru_cell_1/ReadVariableOp_42:
gru_cell_1/ReadVariableOp_5gru_cell_1/ReadVariableOp_52:
gru_cell_1/ReadVariableOp_6gru_cell_1/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_225689

inputs
states_0
readvariableop_resource
readvariableop_1_resource
readvariableop_4_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6Z
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const?
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??I2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_2/Mul_1y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slicem
MatMulMatMulinputsstrided_slice:output:0*
T0*(
_output_shapes
:??????????2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slice_1s
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slice_2s
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack}
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
strided_slice_3|
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddy
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_4/stack}
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_1y
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_2a
mulMulstates_0dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mulg
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_1g
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_2?
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_6t
MatMul_3MatMulmul:z:0strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2

MatMul_3?
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_7v
MatMul_4MatMul	mul_1:z:0strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack}
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
strided_slice_8?
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_3y
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_9/stack}
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
strided_slice_9?
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_4l
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidr
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1?
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_6?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_10w
MatMul_5MatMul	mul_2:z:0strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5{
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
strided_slice_11?
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_5k
mul_3MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
mul_3i
add_2AddV2BiasAdd_2:output:0	mul_3:z:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_4MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
mul_4S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sub[
mul_5Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_5`
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:??????????2
add_3?
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????	:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_6:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?	
?
C__inference_dense_7_layer_call_and_return_conditional_losses_223171

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
C__inference_model_1_layer_call_and_return_conditional_losses_223268

inputs
inputs_1
gru_1_223234
gru_1_223236
gru_1_223238 
layer_normalization_1_223241 
layer_normalization_1_223243
dense_4_223246
dense_4_223248
dense_5_223252
dense_5_223254
dense_6_223257
dense_6_223259
dense_7_223262
dense_7_223264
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_223234gru_1_223236gru_1_223238*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_2227042
gru_1/StatefulPartitionedCall?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0layer_normalization_1_223241layer_normalization_1_223243*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_2230472/
-layer_normalization_1/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_4_223246dense_4_223248*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2230742!
dense_4/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*5J 8? *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_2230972
concatenate_1/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_5_223252dense_5_223254*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2231172!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_223257dense_6_223259*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2231442!
dense_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_223262dense_7_223264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2231712!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^gru_1/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
C__inference_model_1_layer_call_and_return_conditional_losses_223226
input_3
input_4
gru_1_223192
gru_1_223194
gru_1_223196 
layer_normalization_1_223199 
layer_normalization_1_223201
dense_4_223204
dense_4_223206
dense_5_223210
dense_5_223212
dense_6_223215
dense_6_223217
dense_7_223220
dense_7_223222
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallinput_3gru_1_223192gru_1_223194gru_1_223196*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_2229752
gru_1/StatefulPartitionedCall?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0layer_normalization_1_223199layer_normalization_1_223201*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_2230472/
-layer_normalization_1/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_4_223204dense_4_223206*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2230742!
dense_4/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*5J 8? *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_2230972
concatenate_1/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_5_223210dense_5_223212*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2231172!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_223215dense_6_223217*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2231442!
dense_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_223220dense_7_223222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2231712!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall^gru_1/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4
?
?
while_cond_224961
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_224961___redundant_placeholder04
0while_while_cond_224961___redundant_placeholder14
0while_while_cond_224961___redundant_placeholder24
0while_while_cond_224961___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
while_body_224645
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_00
,while_gru_cell_1_readvariableop_1_resource_00
,while_gru_cell_1_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource.
*while_gru_cell_1_readvariableop_1_resource.
*while_gru_cell_1_readvariableop_4_resource??while/gru_cell_1/ReadVariableOp?!while/gru_cell_1/ReadVariableOp_1?!while/gru_cell_1/ReadVariableOp_2?!while/gru_cell_1/ReadVariableOp_3?!while/gru_cell_1/ReadVariableOp_4?!while/gru_cell_1/ReadVariableOp_5?!while/gru_cell_1/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_1/ones_like/Shape?
 while/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_1/ones_like/Const?
while/gru_cell_1/ones_likeFill)while/gru_cell_1/ones_like/Shape:output:0)while/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/ones_like?
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_1/ReadVariableOp?
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_1/unstack?
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_1?
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_1/strided_slice/stack?
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice/stack_1?
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_1/strided_slice/stack_2?
while/gru_cell_1/strided_sliceStridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_1/strided_slice?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul?
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_2?
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_1/stack?
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_1/stack_1?
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_1/stack_2?
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_1?
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_1?
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_3?
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_1/strided_slice_2/stack?
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_1/strided_slice_2/stack_1?
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_2/stack_2?
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_2?
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_2?
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_3/stack?
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_3/stack_1?
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_3/stack_2?
 while/gru_cell_1/strided_slice_3StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_3?
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd?
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_4/stack?
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_4/stack_1?
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_4/stack_2?
 while/gru_cell_1/strided_slice_4StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_4?
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_1?
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_5/stack?
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_1/strided_slice_5/stack_1?
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_5/stack_2?
 while/gru_cell_1/strided_slice_5StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_1/strided_slice_5?
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_2?
while/gru_cell_1/mulMulwhile_placeholder_2#while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul?
while/gru_cell_1/mul_1Mulwhile_placeholder_2#while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_1?
while/gru_cell_1/mul_2Mulwhile_placeholder_2#while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_2?
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_4?
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_1/strided_slice_6/stack?
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_1/strided_slice_6/stack_1?
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_6/stack_2?
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_6?
while/gru_cell_1/MatMul_3MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_3?
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_5?
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_7/stack?
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_7/stack_1?
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_7/stack_2?
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_7?
while/gru_cell_1/MatMul_4MatMulwhile/gru_cell_1/mul_1:z:0)while/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_4?
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_8/stack?
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_8/stack_1?
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_8/stack_2?
 while/gru_cell_1/strided_slice_8StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_8?
while/gru_cell_1/BiasAdd_3BiasAdd#while/gru_cell_1/MatMul_3:product:0)while/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_3?
&while/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_9/stack?
(while/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_9/stack_1?
(while/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_9/stack_2?
 while/gru_cell_1/strided_slice_9StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_9/stack:output:01while/gru_cell_1/strided_slice_9/stack_1:output:01while/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_9?
while/gru_cell_1/BiasAdd_4BiasAdd#while/gru_cell_1/MatMul_4:product:0)while/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_4?
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add?
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid?
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_1?
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid_1?
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_6?
'while/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_1/strided_slice_10/stack?
)while/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_1/strided_slice_10/stack_1?
)while/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_1/strided_slice_10/stack_2?
!while/gru_cell_1/strided_slice_10StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:00while/gru_cell_1/strided_slice_10/stack:output:02while/gru_cell_1/strided_slice_10/stack_1:output:02while/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_1/strided_slice_10?
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul_2:z:0*while/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_5?
'while/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_1/strided_slice_11/stack?
)while/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_1/strided_slice_11/stack_1?
)while/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_1/strided_slice_11/stack_2?
!while/gru_cell_1/strided_slice_11StridedSlice!while/gru_cell_1/unstack:output:10while/gru_cell_1/strided_slice_11/stack:output:02while/gru_cell_1/strided_slice_11/stack_1:output:02while/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_1/strided_slice_11?
while/gru_cell_1/BiasAdd_5BiasAdd#while/gru_cell_1/MatMul_5:product:0*while/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_5?
while/gru_cell_1/mul_3Mulwhile/gru_cell_1/Sigmoid_1:y:0#while/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_3?
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0while/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_2?
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Tanh?
while/gru_cell_1/mul_4Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_4u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_1/sub/x?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/sub?
while/gru_cell_1/mul_5Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_5?
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_4:z:0while/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_1_readvariableop_1_resource,while_gru_cell_1_readvariableop_1_resource_0"Z
*while_gru_cell_1_readvariableop_4_resource,while_gru_cell_1_readvariableop_4_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
Z
.__inference_concatenate_1_layer_call_fn_225509
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*5J 8? *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_2230972
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?	
?
C__inference_dense_4_layer_call_and_return_conditional_losses_223074

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Selu?
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?

C__inference_model_1_layer_call_and_return_conditional_losses_224137
inputs_0
inputs_1,
(gru_1_gru_cell_1_readvariableop_resource.
*gru_1_gru_cell_1_readvariableop_1_resource.
*gru_1_gru_cell_1_readvariableop_4_resource7
3layer_normalization_1_mul_2_readvariableop_resource5
1layer_normalization_1_add_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?gru_1/gru_cell_1/ReadVariableOp?!gru_1/gru_cell_1/ReadVariableOp_1?!gru_1/gru_cell_1/ReadVariableOp_2?!gru_1/gru_cell_1/ReadVariableOp_3?!gru_1/gru_cell_1/ReadVariableOp_4?!gru_1/gru_cell_1/ReadVariableOp_5?!gru_1/gru_cell_1/ReadVariableOp_6?gru_1/while?(layer_normalization_1/add/ReadVariableOp?*layer_normalization_1/mul_2/ReadVariableOpR
gru_1/ShapeShapeinputs_0*
T0*
_output_shapes
:2
gru_1/Shape?
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice/stack?
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_1?
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_2?
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_1/strided_slicei
gru_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_1/zeros/mul/y?
gru_1/zeros/mulMulgru_1/strided_slice:output:0gru_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/mulk
gru_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_1/zeros/Less/y
gru_1/zeros/LessLessgru_1/zeros/mul:z:0gru_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/Lesso
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_1/zeros/packed/1?
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_1/zeros/packedk
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_1/zeros/Const?
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_1/zeros?
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_1/transpose/perm?
gru_1/transpose	Transposeinputs_0gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????	2
gru_1/transposea
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
:2
gru_1/Shape_1?
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_1/stack?
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_1/stack_1?
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_1/stack_2?
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_1/strided_slice_1?
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru_1/TensorArrayV2/element_shape?
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_1/TensorArrayV2?
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2=
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_1/TensorArrayUnstack/TensorListFromTensor?
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_2/stack?
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_2/stack_1?
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_2/stack_2?
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
gru_1/strided_slice_2?
 gru_1/gru_cell_1/ones_like/ShapeShapegru_1/zeros:output:0*
T0*
_output_shapes
:2"
 gru_1/gru_cell_1/ones_like/Shape?
 gru_1/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 gru_1/gru_cell_1/ones_like/Const?
gru_1/gru_cell_1/ones_likeFill)gru_1/gru_cell_1/ones_like/Shape:output:0)gru_1/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/ones_like?
gru_1/gru_cell_1/ReadVariableOpReadVariableOp(gru_1_gru_cell_1_readvariableop_resource*
_output_shapes
:	?*
dtype02!
gru_1/gru_cell_1/ReadVariableOp?
gru_1/gru_cell_1/unstackUnpack'gru_1/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_1/gru_cell_1/unstack?
!gru_1/gru_cell_1/ReadVariableOp_1ReadVariableOp*gru_1_gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_1?
$gru_1/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$gru_1/gru_cell_1/strided_slice/stack?
&gru_1/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_1/gru_cell_1/strided_slice/stack_1?
&gru_1/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru_1/gru_cell_1/strided_slice/stack_2?
gru_1/gru_cell_1/strided_sliceStridedSlice)gru_1/gru_cell_1/ReadVariableOp_1:value:0-gru_1/gru_cell_1/strided_slice/stack:output:0/gru_1/gru_cell_1/strided_slice/stack_1:output:0/gru_1/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
gru_1/gru_cell_1/strided_slice?
gru_1/gru_cell_1/MatMulMatMulgru_1/strided_slice_2:output:0'gru_1/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul?
!gru_1/gru_cell_1/ReadVariableOp_2ReadVariableOp*gru_1_gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_2?
&gru_1/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_1/gru_cell_1/strided_slice_1/stack?
(gru_1/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(gru_1/gru_cell_1/strided_slice_1/stack_1?
(gru_1/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_1/gru_cell_1/strided_slice_1/stack_2?
 gru_1/gru_cell_1/strided_slice_1StridedSlice)gru_1/gru_cell_1/ReadVariableOp_2:value:0/gru_1/gru_cell_1/strided_slice_1/stack:output:01gru_1/gru_cell_1/strided_slice_1/stack_1:output:01gru_1/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 gru_1/gru_cell_1/strided_slice_1?
gru_1/gru_cell_1/MatMul_1MatMulgru_1/strided_slice_2:output:0)gru_1/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul_1?
!gru_1/gru_cell_1/ReadVariableOp_3ReadVariableOp*gru_1_gru_cell_1_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_3?
&gru_1/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&gru_1/gru_cell_1/strided_slice_2/stack?
(gru_1/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(gru_1/gru_cell_1/strided_slice_2/stack_1?
(gru_1/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_1/gru_cell_1/strided_slice_2/stack_2?
 gru_1/gru_cell_1/strided_slice_2StridedSlice)gru_1/gru_cell_1/ReadVariableOp_3:value:0/gru_1/gru_cell_1/strided_slice_2/stack:output:01gru_1/gru_cell_1/strided_slice_2/stack_1:output:01gru_1/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 gru_1/gru_cell_1/strided_slice_2?
gru_1/gru_cell_1/MatMul_2MatMulgru_1/strided_slice_2:output:0)gru_1/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul_2?
&gru_1/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&gru_1/gru_cell_1/strided_slice_3/stack?
(gru_1/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_1/gru_cell_1/strided_slice_3/stack_1?
(gru_1/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_1/gru_cell_1/strided_slice_3/stack_2?
 gru_1/gru_cell_1/strided_slice_3StridedSlice!gru_1/gru_cell_1/unstack:output:0/gru_1/gru_cell_1/strided_slice_3/stack:output:01gru_1/gru_cell_1/strided_slice_3/stack_1:output:01gru_1/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 gru_1/gru_cell_1/strided_slice_3?
gru_1/gru_cell_1/BiasAddBiasAdd!gru_1/gru_cell_1/MatMul:product:0)gru_1/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd?
&gru_1/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_1/gru_cell_1/strided_slice_4/stack?
(gru_1/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_1/gru_cell_1/strided_slice_4/stack_1?
(gru_1/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_1/gru_cell_1/strided_slice_4/stack_2?
 gru_1/gru_cell_1/strided_slice_4StridedSlice!gru_1/gru_cell_1/unstack:output:0/gru_1/gru_cell_1/strided_slice_4/stack:output:01gru_1/gru_cell_1/strided_slice_4/stack_1:output:01gru_1/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 gru_1/gru_cell_1/strided_slice_4?
gru_1/gru_cell_1/BiasAdd_1BiasAdd#gru_1/gru_cell_1/MatMul_1:product:0)gru_1/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd_1?
&gru_1/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_1/gru_cell_1/strided_slice_5/stack?
(gru_1/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(gru_1/gru_cell_1/strided_slice_5/stack_1?
(gru_1/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_1/gru_cell_1/strided_slice_5/stack_2?
 gru_1/gru_cell_1/strided_slice_5StridedSlice!gru_1/gru_cell_1/unstack:output:0/gru_1/gru_cell_1/strided_slice_5/stack:output:01gru_1/gru_cell_1/strided_slice_5/stack_1:output:01gru_1/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 gru_1/gru_cell_1/strided_slice_5?
gru_1/gru_cell_1/BiasAdd_2BiasAdd#gru_1/gru_cell_1/MatMul_2:product:0)gru_1/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd_2?
gru_1/gru_cell_1/mulMulgru_1/zeros:output:0#gru_1/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul?
gru_1/gru_cell_1/mul_1Mulgru_1/zeros:output:0#gru_1/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul_1?
gru_1/gru_cell_1/mul_2Mulgru_1/zeros:output:0#gru_1/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul_2?
!gru_1/gru_cell_1/ReadVariableOp_4ReadVariableOp*gru_1_gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_4?
&gru_1/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru_1/gru_cell_1/strided_slice_6/stack?
(gru_1/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(gru_1/gru_cell_1/strided_slice_6/stack_1?
(gru_1/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_1/gru_cell_1/strided_slice_6/stack_2?
 gru_1/gru_cell_1/strided_slice_6StridedSlice)gru_1/gru_cell_1/ReadVariableOp_4:value:0/gru_1/gru_cell_1/strided_slice_6/stack:output:01gru_1/gru_cell_1/strided_slice_6/stack_1:output:01gru_1/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 gru_1/gru_cell_1/strided_slice_6?
gru_1/gru_cell_1/MatMul_3MatMulgru_1/gru_cell_1/mul:z:0)gru_1/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul_3?
!gru_1/gru_cell_1/ReadVariableOp_5ReadVariableOp*gru_1_gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_5?
&gru_1/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_1/gru_cell_1/strided_slice_7/stack?
(gru_1/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(gru_1/gru_cell_1/strided_slice_7/stack_1?
(gru_1/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_1/gru_cell_1/strided_slice_7/stack_2?
 gru_1/gru_cell_1/strided_slice_7StridedSlice)gru_1/gru_cell_1/ReadVariableOp_5:value:0/gru_1/gru_cell_1/strided_slice_7/stack:output:01gru_1/gru_cell_1/strided_slice_7/stack_1:output:01gru_1/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 gru_1/gru_cell_1/strided_slice_7?
gru_1/gru_cell_1/MatMul_4MatMulgru_1/gru_cell_1/mul_1:z:0)gru_1/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul_4?
&gru_1/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&gru_1/gru_cell_1/strided_slice_8/stack?
(gru_1/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_1/gru_cell_1/strided_slice_8/stack_1?
(gru_1/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_1/gru_cell_1/strided_slice_8/stack_2?
 gru_1/gru_cell_1/strided_slice_8StridedSlice!gru_1/gru_cell_1/unstack:output:1/gru_1/gru_cell_1/strided_slice_8/stack:output:01gru_1/gru_cell_1/strided_slice_8/stack_1:output:01gru_1/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 gru_1/gru_cell_1/strided_slice_8?
gru_1/gru_cell_1/BiasAdd_3BiasAdd#gru_1/gru_cell_1/MatMul_3:product:0)gru_1/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd_3?
&gru_1/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_1/gru_cell_1/strided_slice_9/stack?
(gru_1/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_1/gru_cell_1/strided_slice_9/stack_1?
(gru_1/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_1/gru_cell_1/strided_slice_9/stack_2?
 gru_1/gru_cell_1/strided_slice_9StridedSlice!gru_1/gru_cell_1/unstack:output:1/gru_1/gru_cell_1/strided_slice_9/stack:output:01gru_1/gru_cell_1/strided_slice_9/stack_1:output:01gru_1/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 gru_1/gru_cell_1/strided_slice_9?
gru_1/gru_cell_1/BiasAdd_4BiasAdd#gru_1/gru_cell_1/MatMul_4:product:0)gru_1/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd_4?
gru_1/gru_cell_1/addAddV2!gru_1/gru_cell_1/BiasAdd:output:0#gru_1/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/add?
gru_1/gru_cell_1/SigmoidSigmoidgru_1/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/Sigmoid?
gru_1/gru_cell_1/add_1AddV2#gru_1/gru_cell_1/BiasAdd_1:output:0#gru_1/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/add_1?
gru_1/gru_cell_1/Sigmoid_1Sigmoidgru_1/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/Sigmoid_1?
!gru_1/gru_cell_1/ReadVariableOp_6ReadVariableOp*gru_1_gru_cell_1_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_1/gru_cell_1/ReadVariableOp_6?
'gru_1/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'gru_1/gru_cell_1/strided_slice_10/stack?
)gru_1/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)gru_1/gru_cell_1/strided_slice_10/stack_1?
)gru_1/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)gru_1/gru_cell_1/strided_slice_10/stack_2?
!gru_1/gru_cell_1/strided_slice_10StridedSlice)gru_1/gru_cell_1/ReadVariableOp_6:value:00gru_1/gru_cell_1/strided_slice_10/stack:output:02gru_1/gru_cell_1/strided_slice_10/stack_1:output:02gru_1/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!gru_1/gru_cell_1/strided_slice_10?
gru_1/gru_cell_1/MatMul_5MatMulgru_1/gru_cell_1/mul_2:z:0*gru_1/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/MatMul_5?
'gru_1/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'gru_1/gru_cell_1/strided_slice_11/stack?
)gru_1/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)gru_1/gru_cell_1/strided_slice_11/stack_1?
)gru_1/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)gru_1/gru_cell_1/strided_slice_11/stack_2?
!gru_1/gru_cell_1/strided_slice_11StridedSlice!gru_1/gru_cell_1/unstack:output:10gru_1/gru_cell_1/strided_slice_11/stack:output:02gru_1/gru_cell_1/strided_slice_11/stack_1:output:02gru_1/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!gru_1/gru_cell_1/strided_slice_11?
gru_1/gru_cell_1/BiasAdd_5BiasAdd#gru_1/gru_cell_1/MatMul_5:product:0*gru_1/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/BiasAdd_5?
gru_1/gru_cell_1/mul_3Mulgru_1/gru_cell_1/Sigmoid_1:y:0#gru_1/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul_3?
gru_1/gru_cell_1/add_2AddV2#gru_1/gru_cell_1/BiasAdd_2:output:0gru_1/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/add_2?
gru_1/gru_cell_1/TanhTanhgru_1/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/Tanh?
gru_1/gru_cell_1/mul_4Mulgru_1/gru_cell_1/Sigmoid:y:0gru_1/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul_4u
gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_1/gru_cell_1/sub/x?
gru_1/gru_cell_1/subSubgru_1/gru_cell_1/sub/x:output:0gru_1/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/sub?
gru_1/gru_cell_1/mul_5Mulgru_1/gru_cell_1/sub:z:0gru_1/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/mul_5?
gru_1/gru_cell_1/add_3AddV2gru_1/gru_cell_1/mul_4:z:0gru_1/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_1/gru_cell_1/add_3?
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2%
#gru_1/TensorArrayV2_1/element_shape?
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_1/TensorArrayV2_1Z

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_1/time?
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru_1/while/maximum_iterationsv
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_1/while/loop_counter?
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_1_readvariableop_resource*gru_1_gru_cell_1_readvariableop_1_resource*gru_1_gru_cell_1_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*#
bodyR
gru_1_while_body_223923*#
condR
gru_1_while_cond_223922*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_1/while?
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   28
6gru_1/TensorArrayV2Stack/TensorListStack/element_shape?
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02*
(gru_1/TensorArrayV2Stack/TensorListStack?
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_1/strided_slice_3/stack?
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_3/stack_1?
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_3/stack_2?
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_1/strided_slice_3?
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_1/transpose_1/perm?
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_1/transpose_1r
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_1/runtime?
layer_normalization_1/ShapeShapegru_1/strided_slice_3:output:0*
T0*
_output_shapes
:2
layer_normalization_1/Shape?
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_1/strided_slice/stack?
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_1?
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_2?
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_1/strided_slice|
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul/x?
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul?
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice_1/stack?
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_1?
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_2?
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_1/strided_slice_1?
layer_normalization_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul_1/x?
layer_normalization_1/mul_1Mul&layer_normalization_1/mul_1/x:output:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul_1?
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/0?
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/3?
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul:z:0layer_normalization_1/mul_1:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_1/Reshape/shape?
layer_normalization_1/ReshapeReshapegru_1/strided_slice_3:output:0,layer_normalization_1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_1/Reshape
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_1/Const?
layer_normalization_1/Fill/dimsPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_1/Fill/dims?
layer_normalization_1/FillFill(layer_normalization_1/Fill/dims:output:0$layer_normalization_1/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_1/Fill?
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_1/Const_1?
!layer_normalization_1/Fill_1/dimsPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_1/Fill_1/dims?
layer_normalization_1/Fill_1Fill*layer_normalization_1/Fill_1/dims:output:0&layer_normalization_1/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_1/Fill_1?
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_2?
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_3?
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/Fill:output:0%layer_normalization_1/Fill_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_1/FusedBatchNormV3?
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*(
_output_shapes
:??????????2!
layer_normalization_1/Reshape_1?
*layer_normalization_1/mul_2/ReadVariableOpReadVariableOp3layer_normalization_1_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*layer_normalization_1/mul_2/ReadVariableOp?
layer_normalization_1/mul_2Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_1/mul_2?
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization_1/add/ReadVariableOp?
layer_normalization_1/addAddV2layer_normalization_1/mul_2:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_1/add?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulinputs_1%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAddq
dense_4/SeluSeludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_4/Selux
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2layer_normalization_1/add:z:0dense_4/Selu:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_1/concat?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulconcatenate_1/concat:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/BiasAddq
dense_5/SeluSeludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_5/Selu?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldense_5/Selu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddq
dense_6/SeluSeludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Selu?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Selu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/BiasAddy
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_7/Sigmoid?
IdentityIdentitydense_7/Sigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^gru_1/gru_cell_1/ReadVariableOp"^gru_1/gru_cell_1/ReadVariableOp_1"^gru_1/gru_cell_1/ReadVariableOp_2"^gru_1/gru_cell_1/ReadVariableOp_3"^gru_1/gru_cell_1/ReadVariableOp_4"^gru_1/gru_cell_1/ReadVariableOp_5"^gru_1/gru_cell_1/ReadVariableOp_6^gru_1/while)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2B
gru_1/gru_cell_1/ReadVariableOpgru_1/gru_cell_1/ReadVariableOp2F
!gru_1/gru_cell_1/ReadVariableOp_1!gru_1/gru_cell_1/ReadVariableOp_12F
!gru_1/gru_cell_1/ReadVariableOp_2!gru_1/gru_cell_1/ReadVariableOp_22F
!gru_1/gru_cell_1/ReadVariableOp_3!gru_1/gru_cell_1/ReadVariableOp_32F
!gru_1/gru_cell_1/ReadVariableOp_4!gru_1/gru_cell_1/ReadVariableOp_42F
!gru_1/gru_cell_1/ReadVariableOp_5!gru_1/gru_cell_1/ReadVariableOp_52F
!gru_1/gru_cell_1/ReadVariableOp_6!gru_1/gru_cell_1/ReadVariableOp_62
gru_1/whilegru_1/while2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_2/ReadVariableOp*layer_normalization_1/mul_2/ReadVariableOp:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
while_cond_222828
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_222828___redundant_placeholder04
0while_while_cond_222828___redundant_placeholder14
0while_while_cond_222828___redundant_placeholder24
0while_while_cond_222828___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
while_body_224962
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_00
,while_gru_cell_1_readvariableop_1_resource_00
,while_gru_cell_1_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource.
*while_gru_cell_1_readvariableop_1_resource.
*while_gru_cell_1_readvariableop_4_resource??while/gru_cell_1/ReadVariableOp?!while/gru_cell_1/ReadVariableOp_1?!while/gru_cell_1/ReadVariableOp_2?!while/gru_cell_1/ReadVariableOp_3?!while/gru_cell_1/ReadVariableOp_4?!while/gru_cell_1/ReadVariableOp_5?!while/gru_cell_1/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_1/ones_like/Shape?
 while/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_1/ones_like/Const?
while/gru_cell_1/ones_likeFill)while/gru_cell_1/ones_like/Shape:output:0)while/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/ones_like?
while/gru_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell_1/dropout/Const?
while/gru_cell_1/dropout/MulMul#while/gru_cell_1/ones_like:output:0'while/gru_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/dropout/Mul?
while/gru_cell_1/dropout/ShapeShape#while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell_1/dropout/Shape?
5while/gru_cell_1/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??127
5while/gru_cell_1/dropout/random_uniform/RandomUniform?
'while/gru_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell_1/dropout/GreaterEqual/y?
%while/gru_cell_1/dropout/GreaterEqualGreaterEqual>while/gru_cell_1/dropout/random_uniform/RandomUniform:output:00while/gru_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell_1/dropout/GreaterEqual?
while/gru_cell_1/dropout/CastCast)while/gru_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell_1/dropout/Cast?
while/gru_cell_1/dropout/Mul_1Mul while/gru_cell_1/dropout/Mul:z:0!while/gru_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_1/dropout/Mul_1?
 while/gru_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_1/dropout_1/Const?
while/gru_cell_1/dropout_1/MulMul#while/gru_cell_1/ones_like:output:0)while/gru_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_1/dropout_1/Mul?
 while/gru_cell_1/dropout_1/ShapeShape#while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_1/dropout_1/Shape?
7while/gru_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???29
7while/gru_cell_1/dropout_1/random_uniform/RandomUniform?
)while/gru_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_1/dropout_1/GreaterEqual/y?
'while/gru_cell_1/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_1/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_1/dropout_1/GreaterEqual?
while/gru_cell_1/dropout_1/CastCast+while/gru_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_1/dropout_1/Cast?
 while/gru_cell_1/dropout_1/Mul_1Mul"while/gru_cell_1/dropout_1/Mul:z:0#while/gru_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_1/dropout_1/Mul_1?
 while/gru_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_1/dropout_2/Const?
while/gru_cell_1/dropout_2/MulMul#while/gru_cell_1/ones_like:output:0)while/gru_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_1/dropout_2/Mul?
 while/gru_cell_1/dropout_2/ShapeShape#while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_1/dropout_2/Shape?
7while/gru_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???29
7while/gru_cell_1/dropout_2/random_uniform/RandomUniform?
)while/gru_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_1/dropout_2/GreaterEqual/y?
'while/gru_cell_1/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_1/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_1/dropout_2/GreaterEqual?
while/gru_cell_1/dropout_2/CastCast+while/gru_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_1/dropout_2/Cast?
 while/gru_cell_1/dropout_2/Mul_1Mul"while/gru_cell_1/dropout_2/Mul:z:0#while/gru_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_1/dropout_2/Mul_1?
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_1/ReadVariableOp?
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_1/unstack?
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_1?
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_1/strided_slice/stack?
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice/stack_1?
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_1/strided_slice/stack_2?
while/gru_cell_1/strided_sliceStridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_1/strided_slice?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul?
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_2?
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_1/stack?
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_1/stack_1?
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_1/stack_2?
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_1?
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_1?
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_3?
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_1/strided_slice_2/stack?
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_1/strided_slice_2/stack_1?
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_2/stack_2?
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_2?
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_2?
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_3/stack?
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_3/stack_1?
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_3/stack_2?
 while/gru_cell_1/strided_slice_3StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_3?
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd?
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_4/stack?
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_4/stack_1?
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_4/stack_2?
 while/gru_cell_1/strided_slice_4StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_4?
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_1?
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_5/stack?
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_1/strided_slice_5/stack_1?
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_5/stack_2?
 while/gru_cell_1/strided_slice_5StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_1/strided_slice_5?
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_2?
while/gru_cell_1/mulMulwhile_placeholder_2"while/gru_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul?
while/gru_cell_1/mul_1Mulwhile_placeholder_2$while/gru_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_1?
while/gru_cell_1/mul_2Mulwhile_placeholder_2$while/gru_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_2?
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_4?
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_1/strided_slice_6/stack?
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_1/strided_slice_6/stack_1?
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_6/stack_2?
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_6?
while/gru_cell_1/MatMul_3MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_3?
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_5?
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_7/stack?
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_7/stack_1?
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_7/stack_2?
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_7?
while/gru_cell_1/MatMul_4MatMulwhile/gru_cell_1/mul_1:z:0)while/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_4?
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_8/stack?
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_8/stack_1?
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_8/stack_2?
 while/gru_cell_1/strided_slice_8StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_8?
while/gru_cell_1/BiasAdd_3BiasAdd#while/gru_cell_1/MatMul_3:product:0)while/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_3?
&while/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_9/stack?
(while/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_9/stack_1?
(while/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_9/stack_2?
 while/gru_cell_1/strided_slice_9StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_9/stack:output:01while/gru_cell_1/strided_slice_9/stack_1:output:01while/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_9?
while/gru_cell_1/BiasAdd_4BiasAdd#while/gru_cell_1/MatMul_4:product:0)while/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_4?
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add?
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid?
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_1?
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid_1?
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_6?
'while/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_1/strided_slice_10/stack?
)while/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_1/strided_slice_10/stack_1?
)while/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_1/strided_slice_10/stack_2?
!while/gru_cell_1/strided_slice_10StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:00while/gru_cell_1/strided_slice_10/stack:output:02while/gru_cell_1/strided_slice_10/stack_1:output:02while/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_1/strided_slice_10?
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul_2:z:0*while/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_5?
'while/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_1/strided_slice_11/stack?
)while/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_1/strided_slice_11/stack_1?
)while/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_1/strided_slice_11/stack_2?
!while/gru_cell_1/strided_slice_11StridedSlice!while/gru_cell_1/unstack:output:10while/gru_cell_1/strided_slice_11/stack:output:02while/gru_cell_1/strided_slice_11/stack_1:output:02while/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_1/strided_slice_11?
while/gru_cell_1/BiasAdd_5BiasAdd#while/gru_cell_1/MatMul_5:product:0*while/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_5?
while/gru_cell_1/mul_3Mulwhile/gru_cell_1/Sigmoid_1:y:0#while/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_3?
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0while/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_2?
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Tanh?
while/gru_cell_1/mul_4Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_4u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_1/sub/x?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/sub?
while/gru_cell_1/mul_5Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_5?
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_4:z:0while/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_1_readvariableop_1_resource,while_gru_cell_1_readvariableop_1_resource_0"Z
*while_gru_cell_1_readvariableop_4_resource,while_gru_cell_1_readvariableop_4_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_222533
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_222533___redundant_placeholder04
0while_while_cond_222533___redundant_placeholder14
0while_while_cond_222533___redundant_placeholder24
0while_while_cond_222533___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
C__inference_dense_5_layer_call_and_return_conditional_losses_225520

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Selu?
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
(__inference_model_1_layer_call_fn_224201
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2233382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:?????????	:?????????:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?!
?
while_body_222190
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_1_222212_0
while_gru_cell_1_222214_0
while_gru_cell_1_222216_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_1_222212
while_gru_cell_1_222214
while_gru_cell_1_222216??(while/gru_cell_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_222212_0while_gru_cell_1_222214_0while_gru_cell_1_222216_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_2218352*
(while/gru_cell_1/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1)^while/gru_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4"4
while_gru_cell_1_222212while_gru_cell_1_222212_0"4
while_gru_cell_1_222214while_gru_cell_1_222214_0"4
while_gru_cell_1_222216while_gru_cell_1_222216_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2T
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
}
(__inference_dense_6_layer_call_fn_225549

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2231442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
+__inference_gru_cell_1_layer_call_fn_225813

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*5J 8? *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_2219312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????	:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?h
?
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_225785

inputs
states_0
readvariableop_resource
readvariableop_1_resource
readvariableop_4_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6Z
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_likey
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slicem
MatMulMatMulinputsstrided_slice:output:0*
T0*(
_output_shapes
:??????????2
MatMul
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slice_1s
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_1
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:		?*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
strided_slice_2s
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack}
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
strided_slice_3|
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddy
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_4/stack}
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_1y
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_2b
mulMulstates_0ones_like:output:0*
T0*(
_output_shapes
:??????????2
mulf
mul_1Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_1f
mul_2Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_2?
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_6t
MatMul_3MatMulmul:z:0strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2

MatMul_3?
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_7v
MatMul_4MatMul	mul_1:z:0strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack}
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
strided_slice_8?
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_3y
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_9/stack}
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
strided_slice_9?
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_4l
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidr
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1?
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_6?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_10w
MatMul_5MatMul	mul_2:z:0strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5{
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
strided_slice_11?
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
	BiasAdd_5k
mul_3MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
mul_3i
add_2AddV2BiasAdd_2:output:0	mul_3:z:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_4MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
mul_4S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sub[
mul_5Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_5`
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:??????????2
add_3?
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????	:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_6:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
u
I__inference_concatenate_1_layer_call_and_return_conditional_losses_225503
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?i
?
__inference__traced_save_225990
file_prefix:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop6
2savev2_gru_1_gru_cell_1_kernel_read_readvariableop@
<savev2_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop4
0savev2_gru_1_gru_cell_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableopB
>savev2_nadam_layer_normalization_1_gamma_m_read_readvariableopA
=savev2_nadam_layer_normalization_1_beta_m_read_readvariableop5
1savev2_nadam_dense_4_kernel_m_read_readvariableop3
/savev2_nadam_dense_4_bias_m_read_readvariableop5
1savev2_nadam_dense_5_kernel_m_read_readvariableop3
/savev2_nadam_dense_5_bias_m_read_readvariableop5
1savev2_nadam_dense_6_kernel_m_read_readvariableop3
/savev2_nadam_dense_6_bias_m_read_readvariableop5
1savev2_nadam_dense_7_kernel_m_read_readvariableop3
/savev2_nadam_dense_7_bias_m_read_readvariableop>
:savev2_nadam_gru_1_gru_cell_1_kernel_m_read_readvariableopH
Dsavev2_nadam_gru_1_gru_cell_1_recurrent_kernel_m_read_readvariableop<
8savev2_nadam_gru_1_gru_cell_1_bias_m_read_readvariableopB
>savev2_nadam_layer_normalization_1_gamma_v_read_readvariableopA
=savev2_nadam_layer_normalization_1_beta_v_read_readvariableop5
1savev2_nadam_dense_4_kernel_v_read_readvariableop3
/savev2_nadam_dense_4_bias_v_read_readvariableop5
1savev2_nadam_dense_5_kernel_v_read_readvariableop3
/savev2_nadam_dense_5_bias_v_read_readvariableop5
1savev2_nadam_dense_6_kernel_v_read_readvariableop3
/savev2_nadam_dense_6_bias_v_read_readvariableop5
1savev2_nadam_dense_7_kernel_v_read_readvariableop3
/savev2_nadam_dense_7_bias_v_read_readvariableop>
:savev2_nadam_gru_1_gru_cell_1_kernel_v_read_readvariableopH
Dsavev2_nadam_gru_1_gru_cell_1_recurrent_kernel_v_read_readvariableop<
8savev2_nadam_gru_1_gru_cell_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop2savev2_gru_1_gru_cell_1_kernel_read_readvariableop<savev2_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop0savev2_gru_1_gru_cell_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop>savev2_nadam_layer_normalization_1_gamma_m_read_readvariableop=savev2_nadam_layer_normalization_1_beta_m_read_readvariableop1savev2_nadam_dense_4_kernel_m_read_readvariableop/savev2_nadam_dense_4_bias_m_read_readvariableop1savev2_nadam_dense_5_kernel_m_read_readvariableop/savev2_nadam_dense_5_bias_m_read_readvariableop1savev2_nadam_dense_6_kernel_m_read_readvariableop/savev2_nadam_dense_6_bias_m_read_readvariableop1savev2_nadam_dense_7_kernel_m_read_readvariableop/savev2_nadam_dense_7_bias_m_read_readvariableop:savev2_nadam_gru_1_gru_cell_1_kernel_m_read_readvariableopDsavev2_nadam_gru_1_gru_cell_1_recurrent_kernel_m_read_readvariableop8savev2_nadam_gru_1_gru_cell_1_bias_m_read_readvariableop>savev2_nadam_layer_normalization_1_gamma_v_read_readvariableop=savev2_nadam_layer_normalization_1_beta_v_read_readvariableop1savev2_nadam_dense_4_kernel_v_read_readvariableop/savev2_nadam_dense_4_bias_v_read_readvariableop1savev2_nadam_dense_5_kernel_v_read_readvariableop/savev2_nadam_dense_5_bias_v_read_readvariableop1savev2_nadam_dense_6_kernel_v_read_readvariableop/savev2_nadam_dense_6_bias_v_read_readvariableop1savev2_nadam_dense_7_kernel_v_read_readvariableop/savev2_nadam_dense_7_bias_v_read_readvariableop:savev2_nadam_gru_1_gru_cell_1_kernel_v_read_readvariableopDsavev2_nadam_gru_1_gru_cell_1_recurrent_kernel_v_read_readvariableop8savev2_nadam_gru_1_gru_cell_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?:	?:?:
??:?:
??:?:	?:: : : : : : :		?:
??:	?: : :?:?:?:?:?:?:	?:?:
??:?:
??:?:	?::		?:
??:	?:?:?:	?:?:
??:?:
??:?:	?::		?:
??:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%	!

_output_shapes
:	?: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:& "
 
_output_shapes
:
??:!!

_output_shapes	
:?:%"!

_output_shapes
:	?: #

_output_shapes
::%$!

_output_shapes
:		?:&%"
 
_output_shapes
:
??:%&!

_output_shapes
:	?:!'

_output_shapes	
:?:!(

_output_shapes	
:?:%)!

_output_shapes
:	?:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:%/!

_output_shapes
:	?: 0

_output_shapes
::%1!

_output_shapes
:		?:&2"
 
_output_shapes
:
??:%3!

_output_shapes
:	?:4

_output_shapes
: 
??
?
while_body_222829
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_00
,while_gru_cell_1_readvariableop_1_resource_00
,while_gru_cell_1_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource.
*while_gru_cell_1_readvariableop_1_resource.
*while_gru_cell_1_readvariableop_4_resource??while/gru_cell_1/ReadVariableOp?!while/gru_cell_1/ReadVariableOp_1?!while/gru_cell_1/ReadVariableOp_2?!while/gru_cell_1/ReadVariableOp_3?!while/gru_cell_1/ReadVariableOp_4?!while/gru_cell_1/ReadVariableOp_5?!while/gru_cell_1/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_1/ones_like/Shape?
 while/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_1/ones_like/Const?
while/gru_cell_1/ones_likeFill)while/gru_cell_1/ones_like/Shape:output:0)while/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/ones_like?
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_1/ReadVariableOp?
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_1/unstack?
!while/gru_cell_1/ReadVariableOp_1ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_1?
$while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_1/strided_slice/stack?
&while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice/stack_1?
&while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_1/strided_slice/stack_2?
while/gru_cell_1/strided_sliceStridedSlice)while/gru_cell_1/ReadVariableOp_1:value:0-while/gru_cell_1/strided_slice/stack:output:0/while/gru_cell_1/strided_slice/stack_1:output:0/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_1/strided_slice?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul?
!while/gru_cell_1/ReadVariableOp_2ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_2?
&while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_1/stack?
(while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_1/stack_1?
(while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_1/stack_2?
 while/gru_cell_1/strided_slice_1StridedSlice)while/gru_cell_1/ReadVariableOp_2:value:0/while/gru_cell_1/strided_slice_1/stack:output:01while/gru_cell_1/strided_slice_1/stack_1:output:01while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_1?
while/gru_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_1?
!while/gru_cell_1/ReadVariableOp_3ReadVariableOp,while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_1/ReadVariableOp_3?
&while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_1/strided_slice_2/stack?
(while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_1/strided_slice_2/stack_1?
(while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_2/stack_2?
 while/gru_cell_1/strided_slice_2StridedSlice)while/gru_cell_1/ReadVariableOp_3:value:0/while/gru_cell_1/strided_slice_2/stack:output:01while/gru_cell_1/strided_slice_2/stack_1:output:01while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_2?
while/gru_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_2?
&while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_3/stack?
(while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_3/stack_1?
(while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_3/stack_2?
 while/gru_cell_1/strided_slice_3StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_3/stack:output:01while/gru_cell_1/strided_slice_3/stack_1:output:01while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_3?
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0)while/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd?
&while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_4/stack?
(while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_4/stack_1?
(while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_4/stack_2?
 while/gru_cell_1/strided_slice_4StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_4/stack:output:01while/gru_cell_1/strided_slice_4/stack_1:output:01while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_4?
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0)while/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_1?
&while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_5/stack?
(while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_1/strided_slice_5/stack_1?
(while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_5/stack_2?
 while/gru_cell_1/strided_slice_5StridedSlice!while/gru_cell_1/unstack:output:0/while/gru_cell_1/strided_slice_5/stack:output:01while/gru_cell_1/strided_slice_5/stack_1:output:01while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_1/strided_slice_5?
while/gru_cell_1/BiasAdd_2BiasAdd#while/gru_cell_1/MatMul_2:product:0)while/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_2?
while/gru_cell_1/mulMulwhile_placeholder_2#while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul?
while/gru_cell_1/mul_1Mulwhile_placeholder_2#while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_1?
while/gru_cell_1/mul_2Mulwhile_placeholder_2#while/gru_cell_1/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_2?
!while/gru_cell_1/ReadVariableOp_4ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_4?
&while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_1/strided_slice_6/stack?
(while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_1/strided_slice_6/stack_1?
(while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_6/stack_2?
 while/gru_cell_1/strided_slice_6StridedSlice)while/gru_cell_1/ReadVariableOp_4:value:0/while/gru_cell_1/strided_slice_6/stack:output:01while/gru_cell_1/strided_slice_6/stack_1:output:01while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_6?
while/gru_cell_1/MatMul_3MatMulwhile/gru_cell_1/mul:z:0)while/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_3?
!while/gru_cell_1/ReadVariableOp_5ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_5?
&while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_1/strided_slice_7/stack?
(while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_1/strided_slice_7/stack_1?
(while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_1/strided_slice_7/stack_2?
 while/gru_cell_1/strided_slice_7StridedSlice)while/gru_cell_1/ReadVariableOp_5:value:0/while/gru_cell_1/strided_slice_7/stack:output:01while/gru_cell_1/strided_slice_7/stack_1:output:01while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_1/strided_slice_7?
while/gru_cell_1/MatMul_4MatMulwhile/gru_cell_1/mul_1:z:0)while/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_4?
&while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_1/strided_slice_8/stack?
(while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_8/stack_1?
(while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_8/stack_2?
 while/gru_cell_1/strided_slice_8StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_8/stack:output:01while/gru_cell_1/strided_slice_8/stack_1:output:01while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_1/strided_slice_8?
while/gru_cell_1/BiasAdd_3BiasAdd#while/gru_cell_1/MatMul_3:product:0)while/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_3?
&while/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_1/strided_slice_9/stack?
(while/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_1/strided_slice_9/stack_1?
(while/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_1/strided_slice_9/stack_2?
 while/gru_cell_1/strided_slice_9StridedSlice!while/gru_cell_1/unstack:output:1/while/gru_cell_1/strided_slice_9/stack:output:01while/gru_cell_1/strided_slice_9/stack_1:output:01while/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_1/strided_slice_9?
while/gru_cell_1/BiasAdd_4BiasAdd#while/gru_cell_1/MatMul_4:product:0)while/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_4?
while/gru_cell_1/addAddV2!while/gru_cell_1/BiasAdd:output:0#while/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add?
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid?
while/gru_cell_1/add_1AddV2#while/gru_cell_1/BiasAdd_1:output:0#while/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_1?
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Sigmoid_1?
!while/gru_cell_1/ReadVariableOp_6ReadVariableOp,while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_1/ReadVariableOp_6?
'while/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_1/strided_slice_10/stack?
)while/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_1/strided_slice_10/stack_1?
)while/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_1/strided_slice_10/stack_2?
!while/gru_cell_1/strided_slice_10StridedSlice)while/gru_cell_1/ReadVariableOp_6:value:00while/gru_cell_1/strided_slice_10/stack:output:02while/gru_cell_1/strided_slice_10/stack_1:output:02while/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_1/strided_slice_10?
while/gru_cell_1/MatMul_5MatMulwhile/gru_cell_1/mul_2:z:0*while/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/MatMul_5?
'while/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_1/strided_slice_11/stack?
)while/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_1/strided_slice_11/stack_1?
)while/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_1/strided_slice_11/stack_2?
!while/gru_cell_1/strided_slice_11StridedSlice!while/gru_cell_1/unstack:output:10while/gru_cell_1/strided_slice_11/stack:output:02while/gru_cell_1/strided_slice_11/stack_1:output:02while/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_1/strided_slice_11?
while/gru_cell_1/BiasAdd_5BiasAdd#while/gru_cell_1/MatMul_5:product:0*while/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/BiasAdd_5?
while/gru_cell_1/mul_3Mulwhile/gru_cell_1/Sigmoid_1:y:0#while/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_3?
while/gru_cell_1/add_2AddV2#while/gru_cell_1/BiasAdd_2:output:0while/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_2?
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/Tanh?
while/gru_cell_1/mul_4Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_4u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_1/sub/x?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/sub?
while/gru_cell_1/mul_5Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/mul_5?
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_4:z:0while/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_1/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0 ^while/gru_cell_1/ReadVariableOp"^while/gru_cell_1/ReadVariableOp_1"^while/gru_cell_1/ReadVariableOp_2"^while/gru_cell_1/ReadVariableOp_3"^while/gru_cell_1/ReadVariableOp_4"^while/gru_cell_1/ReadVariableOp_5"^while/gru_cell_1/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_1_readvariableop_1_resource,while_gru_cell_1_readvariableop_1_resource_0"Z
*while_gru_cell_1_readvariableop_4_resource,while_gru_cell_1_readvariableop_4_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp2F
!while/gru_cell_1/ReadVariableOp_1!while/gru_cell_1/ReadVariableOp_12F
!while/gru_cell_1/ReadVariableOp_2!while/gru_cell_1/ReadVariableOp_22F
!while/gru_cell_1/ReadVariableOp_3!while/gru_cell_1/ReadVariableOp_32F
!while/gru_cell_1/ReadVariableOp_4!while/gru_cell_1/ReadVariableOp_42F
!while/gru_cell_1/ReadVariableOp_5!while/gru_cell_1/ReadVariableOp_52F
!while/gru_cell_1/ReadVariableOp_6!while/gru_cell_1/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?	
gru_1_while_body_223559(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_04
0gru_1_while_gru_cell_1_readvariableop_resource_06
2gru_1_while_gru_cell_1_readvariableop_1_resource_06
2gru_1_while_gru_cell_1_readvariableop_4_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor2
.gru_1_while_gru_cell_1_readvariableop_resource4
0gru_1_while_gru_cell_1_readvariableop_1_resource4
0gru_1_while_gru_cell_1_readvariableop_4_resource??%gru_1/while/gru_cell_1/ReadVariableOp?'gru_1/while/gru_cell_1/ReadVariableOp_1?'gru_1/while/gru_cell_1/ReadVariableOp_2?'gru_1/while/gru_cell_1/ReadVariableOp_3?'gru_1/while/gru_cell_1/ReadVariableOp_4?'gru_1/while/gru_cell_1/ReadVariableOp_5?'gru_1/while/gru_cell_1/ReadVariableOp_6?
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2?
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype021
/gru_1/while/TensorArrayV2Read/TensorListGetItem?
&gru_1/while/gru_cell_1/ones_like/ShapeShapegru_1_while_placeholder_2*
T0*
_output_shapes
:2(
&gru_1/while/gru_cell_1/ones_like/Shape?
&gru_1/while/gru_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&gru_1/while/gru_cell_1/ones_like/Const?
 gru_1/while/gru_cell_1/ones_likeFill/gru_1/while/gru_cell_1/ones_like/Shape:output:0/gru_1/while/gru_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/ones_like?
$gru_1/while/gru_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$gru_1/while/gru_cell_1/dropout/Const?
"gru_1/while/gru_cell_1/dropout/MulMul)gru_1/while/gru_cell_1/ones_like:output:0-gru_1/while/gru_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2$
"gru_1/while/gru_cell_1/dropout/Mul?
$gru_1/while/gru_cell_1/dropout/ShapeShape)gru_1/while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2&
$gru_1/while/gru_cell_1/dropout/Shape?
;gru_1/while/gru_cell_1/dropout/random_uniform/RandomUniformRandomUniform-gru_1/while/gru_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ڟ2=
;gru_1/while/gru_cell_1/dropout/random_uniform/RandomUniform?
-gru_1/while/gru_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-gru_1/while/gru_cell_1/dropout/GreaterEqual/y?
+gru_1/while/gru_cell_1/dropout/GreaterEqualGreaterEqualDgru_1/while/gru_cell_1/dropout/random_uniform/RandomUniform:output:06gru_1/while/gru_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2-
+gru_1/while/gru_cell_1/dropout/GreaterEqual?
#gru_1/while/gru_cell_1/dropout/CastCast/gru_1/while/gru_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2%
#gru_1/while/gru_cell_1/dropout/Cast?
$gru_1/while/gru_cell_1/dropout/Mul_1Mul&gru_1/while/gru_cell_1/dropout/Mul:z:0'gru_1/while/gru_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2&
$gru_1/while/gru_cell_1/dropout/Mul_1?
&gru_1/while/gru_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&gru_1/while/gru_cell_1/dropout_1/Const?
$gru_1/while/gru_cell_1/dropout_1/MulMul)gru_1/while/gru_cell_1/ones_like:output:0/gru_1/while/gru_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$gru_1/while/gru_cell_1/dropout_1/Mul?
&gru_1/while/gru_cell_1/dropout_1/ShapeShape)gru_1/while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2(
&gru_1/while/gru_cell_1/dropout_1/Shape?
=gru_1/while/gru_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform/gru_1/while/gru_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=gru_1/while/gru_cell_1/dropout_1/random_uniform/RandomUniform?
/gru_1/while/gru_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/gru_1/while/gru_cell_1/dropout_1/GreaterEqual/y?
-gru_1/while/gru_cell_1/dropout_1/GreaterEqualGreaterEqualFgru_1/while/gru_cell_1/dropout_1/random_uniform/RandomUniform:output:08gru_1/while/gru_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-gru_1/while/gru_cell_1/dropout_1/GreaterEqual?
%gru_1/while/gru_cell_1/dropout_1/CastCast1gru_1/while/gru_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%gru_1/while/gru_cell_1/dropout_1/Cast?
&gru_1/while/gru_cell_1/dropout_1/Mul_1Mul(gru_1/while/gru_cell_1/dropout_1/Mul:z:0)gru_1/while/gru_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&gru_1/while/gru_cell_1/dropout_1/Mul_1?
&gru_1/while/gru_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&gru_1/while/gru_cell_1/dropout_2/Const?
$gru_1/while/gru_cell_1/dropout_2/MulMul)gru_1/while/gru_cell_1/ones_like:output:0/gru_1/while/gru_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2&
$gru_1/while/gru_cell_1/dropout_2/Mul?
&gru_1/while/gru_cell_1/dropout_2/ShapeShape)gru_1/while/gru_cell_1/ones_like:output:0*
T0*
_output_shapes
:2(
&gru_1/while/gru_cell_1/dropout_2/Shape?
=gru_1/while/gru_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform/gru_1/while/gru_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??|2?
=gru_1/while/gru_cell_1/dropout_2/random_uniform/RandomUniform?
/gru_1/while/gru_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/gru_1/while/gru_cell_1/dropout_2/GreaterEqual/y?
-gru_1/while/gru_cell_1/dropout_2/GreaterEqualGreaterEqualFgru_1/while/gru_cell_1/dropout_2/random_uniform/RandomUniform:output:08gru_1/while/gru_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-gru_1/while/gru_cell_1/dropout_2/GreaterEqual?
%gru_1/while/gru_cell_1/dropout_2/CastCast1gru_1/while/gru_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%gru_1/while/gru_cell_1/dropout_2/Cast?
&gru_1/while/gru_cell_1/dropout_2/Mul_1Mul(gru_1/while/gru_cell_1/dropout_2/Mul:z:0)gru_1/while/gru_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&gru_1/while/gru_cell_1/dropout_2/Mul_1?
%gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	?*
dtype02'
%gru_1/while/gru_cell_1/ReadVariableOp?
gru_1/while/gru_cell_1/unstackUnpack-gru_1/while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2 
gru_1/while/gru_cell_1/unstack?
'gru_1/while/gru_cell_1/ReadVariableOp_1ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_1?
*gru_1/while/gru_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*gru_1/while/gru_cell_1/strided_slice/stack?
,gru_1/while/gru_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_1/while/gru_cell_1/strided_slice/stack_1?
,gru_1/while/gru_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru_1/while/gru_cell_1/strided_slice/stack_2?
$gru_1/while/gru_cell_1/strided_sliceStridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_1:value:03gru_1/while/gru_cell_1/strided_slice/stack:output:05gru_1/while/gru_cell_1/strided_slice/stack_1:output:05gru_1/while/gru_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2&
$gru_1/while/gru_cell_1/strided_slice?
gru_1/while/gru_cell_1/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0-gru_1/while/gru_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/MatMul?
'gru_1/while/gru_cell_1/ReadVariableOp_2ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_2?
,gru_1/while/gru_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_1/while/gru_cell_1/strided_slice_1/stack?
.gru_1/while/gru_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  20
.gru_1/while/gru_cell_1/strided_slice_1/stack_1?
.gru_1/while/gru_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_1/while/gru_cell_1/strided_slice_1/stack_2?
&gru_1/while/gru_cell_1/strided_slice_1StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_2:value:05gru_1/while/gru_cell_1/strided_slice_1/stack:output:07gru_1/while/gru_cell_1/strided_slice_1/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&gru_1/while/gru_cell_1/strided_slice_1?
gru_1/while/gru_cell_1/MatMul_1MatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_1/while/gru_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
gru_1/while/gru_cell_1/MatMul_1?
'gru_1/while/gru_cell_1/ReadVariableOp_3ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_3?
,gru_1/while/gru_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2.
,gru_1/while/gru_cell_1/strided_slice_2/stack?
.gru_1/while/gru_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.gru_1/while/gru_cell_1/strided_slice_2/stack_1?
.gru_1/while/gru_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_1/while/gru_cell_1/strided_slice_2/stack_2?
&gru_1/while/gru_cell_1/strided_slice_2StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_3:value:05gru_1/while/gru_cell_1/strided_slice_2/stack:output:07gru_1/while/gru_cell_1/strided_slice_2/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&gru_1/while/gru_cell_1/strided_slice_2?
gru_1/while/gru_cell_1/MatMul_2MatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_1/while/gru_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
gru_1/while/gru_cell_1/MatMul_2?
,gru_1/while/gru_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,gru_1/while/gru_cell_1/strided_slice_3/stack?
.gru_1/while/gru_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_1/while/gru_cell_1/strided_slice_3/stack_1?
.gru_1/while/gru_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_1/while/gru_cell_1/strided_slice_3/stack_2?
&gru_1/while/gru_cell_1/strided_slice_3StridedSlice'gru_1/while/gru_cell_1/unstack:output:05gru_1/while/gru_cell_1/strided_slice_3/stack:output:07gru_1/while/gru_cell_1/strided_slice_3/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2(
&gru_1/while/gru_cell_1/strided_slice_3?
gru_1/while/gru_cell_1/BiasAddBiasAdd'gru_1/while/gru_cell_1/MatMul:product:0/gru_1/while/gru_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2 
gru_1/while/gru_cell_1/BiasAdd?
,gru_1/while/gru_cell_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_1/while/gru_cell_1/strided_slice_4/stack?
.gru_1/while/gru_cell_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_1/while/gru_cell_1/strided_slice_4/stack_1?
.gru_1/while/gru_cell_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_1/while/gru_cell_1/strided_slice_4/stack_2?
&gru_1/while/gru_cell_1/strided_slice_4StridedSlice'gru_1/while/gru_cell_1/unstack:output:05gru_1/while/gru_cell_1/strided_slice_4/stack:output:07gru_1/while/gru_cell_1/strided_slice_4/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2(
&gru_1/while/gru_cell_1/strided_slice_4?
 gru_1/while/gru_cell_1/BiasAdd_1BiasAdd)gru_1/while/gru_cell_1/MatMul_1:product:0/gru_1/while/gru_cell_1/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/BiasAdd_1?
,gru_1/while/gru_cell_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_1/while/gru_cell_1/strided_slice_5/stack?
.gru_1/while/gru_cell_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.gru_1/while/gru_cell_1/strided_slice_5/stack_1?
.gru_1/while/gru_cell_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_1/while/gru_cell_1/strided_slice_5/stack_2?
&gru_1/while/gru_cell_1/strided_slice_5StridedSlice'gru_1/while/gru_cell_1/unstack:output:05gru_1/while/gru_cell_1/strided_slice_5/stack:output:07gru_1/while/gru_cell_1/strided_slice_5/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2(
&gru_1/while/gru_cell_1/strided_slice_5?
 gru_1/while/gru_cell_1/BiasAdd_2BiasAdd)gru_1/while/gru_cell_1/MatMul_2:product:0/gru_1/while/gru_cell_1/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/BiasAdd_2?
gru_1/while/gru_cell_1/mulMulgru_1_while_placeholder_2(gru_1/while/gru_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul?
gru_1/while/gru_cell_1/mul_1Mulgru_1_while_placeholder_2*gru_1/while/gru_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul_1?
gru_1/while/gru_cell_1/mul_2Mulgru_1_while_placeholder_2*gru_1/while/gru_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul_2?
'gru_1/while/gru_cell_1/ReadVariableOp_4ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_4?
,gru_1/while/gru_cell_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,gru_1/while/gru_cell_1/strided_slice_6/stack?
.gru_1/while/gru_cell_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   20
.gru_1/while/gru_cell_1/strided_slice_6/stack_1?
.gru_1/while/gru_cell_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_1/while/gru_cell_1/strided_slice_6/stack_2?
&gru_1/while/gru_cell_1/strided_slice_6StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_4:value:05gru_1/while/gru_cell_1/strided_slice_6/stack:output:07gru_1/while/gru_cell_1/strided_slice_6/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&gru_1/while/gru_cell_1/strided_slice_6?
gru_1/while/gru_cell_1/MatMul_3MatMulgru_1/while/gru_cell_1/mul:z:0/gru_1/while/gru_cell_1/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2!
gru_1/while/gru_cell_1/MatMul_3?
'gru_1/while/gru_cell_1/ReadVariableOp_5ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_5?
,gru_1/while/gru_cell_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_1/while/gru_cell_1/strided_slice_7/stack?
.gru_1/while/gru_cell_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  20
.gru_1/while/gru_cell_1/strided_slice_7/stack_1?
.gru_1/while/gru_cell_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_1/while/gru_cell_1/strided_slice_7/stack_2?
&gru_1/while/gru_cell_1/strided_slice_7StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_5:value:05gru_1/while/gru_cell_1/strided_slice_7/stack:output:07gru_1/while/gru_cell_1/strided_slice_7/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&gru_1/while/gru_cell_1/strided_slice_7?
gru_1/while/gru_cell_1/MatMul_4MatMul gru_1/while/gru_cell_1/mul_1:z:0/gru_1/while/gru_cell_1/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2!
gru_1/while/gru_cell_1/MatMul_4?
,gru_1/while/gru_cell_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,gru_1/while/gru_cell_1/strided_slice_8/stack?
.gru_1/while/gru_cell_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_1/while/gru_cell_1/strided_slice_8/stack_1?
.gru_1/while/gru_cell_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_1/while/gru_cell_1/strided_slice_8/stack_2?
&gru_1/while/gru_cell_1/strided_slice_8StridedSlice'gru_1/while/gru_cell_1/unstack:output:15gru_1/while/gru_cell_1/strided_slice_8/stack:output:07gru_1/while/gru_cell_1/strided_slice_8/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2(
&gru_1/while/gru_cell_1/strided_slice_8?
 gru_1/while/gru_cell_1/BiasAdd_3BiasAdd)gru_1/while/gru_cell_1/MatMul_3:product:0/gru_1/while/gru_cell_1/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/BiasAdd_3?
,gru_1/while/gru_cell_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_1/while/gru_cell_1/strided_slice_9/stack?
.gru_1/while/gru_cell_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_1/while/gru_cell_1/strided_slice_9/stack_1?
.gru_1/while/gru_cell_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_1/while/gru_cell_1/strided_slice_9/stack_2?
&gru_1/while/gru_cell_1/strided_slice_9StridedSlice'gru_1/while/gru_cell_1/unstack:output:15gru_1/while/gru_cell_1/strided_slice_9/stack:output:07gru_1/while/gru_cell_1/strided_slice_9/stack_1:output:07gru_1/while/gru_cell_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2(
&gru_1/while/gru_cell_1/strided_slice_9?
 gru_1/while/gru_cell_1/BiasAdd_4BiasAdd)gru_1/while/gru_cell_1/MatMul_4:product:0/gru_1/while/gru_cell_1/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/BiasAdd_4?
gru_1/while/gru_cell_1/addAddV2'gru_1/while/gru_cell_1/BiasAdd:output:0)gru_1/while/gru_cell_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/add?
gru_1/while/gru_cell_1/SigmoidSigmoidgru_1/while/gru_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2 
gru_1/while/gru_cell_1/Sigmoid?
gru_1/while/gru_cell_1/add_1AddV2)gru_1/while/gru_cell_1/BiasAdd_1:output:0)gru_1/while/gru_cell_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/add_1?
 gru_1/while/gru_cell_1/Sigmoid_1Sigmoid gru_1/while/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/Sigmoid_1?
'gru_1/while/gru_cell_1/ReadVariableOp_6ReadVariableOp2gru_1_while_gru_cell_1_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_1/while/gru_cell_1/ReadVariableOp_6?
-gru_1/while/gru_cell_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2/
-gru_1/while/gru_cell_1/strided_slice_10/stack?
/gru_1/while/gru_cell_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/gru_1/while/gru_cell_1/strided_slice_10/stack_1?
/gru_1/while/gru_cell_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/gru_1/while/gru_cell_1/strided_slice_10/stack_2?
'gru_1/while/gru_cell_1/strided_slice_10StridedSlice/gru_1/while/gru_cell_1/ReadVariableOp_6:value:06gru_1/while/gru_cell_1/strided_slice_10/stack:output:08gru_1/while/gru_cell_1/strided_slice_10/stack_1:output:08gru_1/while/gru_cell_1/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2)
'gru_1/while/gru_cell_1/strided_slice_10?
gru_1/while/gru_cell_1/MatMul_5MatMul gru_1/while/gru_cell_1/mul_2:z:00gru_1/while/gru_cell_1/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2!
gru_1/while/gru_cell_1/MatMul_5?
-gru_1/while/gru_cell_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2/
-gru_1/while/gru_cell_1/strided_slice_11/stack?
/gru_1/while/gru_cell_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/gru_1/while/gru_cell_1/strided_slice_11/stack_1?
/gru_1/while/gru_cell_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/gru_1/while/gru_cell_1/strided_slice_11/stack_2?
'gru_1/while/gru_cell_1/strided_slice_11StridedSlice'gru_1/while/gru_cell_1/unstack:output:16gru_1/while/gru_cell_1/strided_slice_11/stack:output:08gru_1/while/gru_cell_1/strided_slice_11/stack_1:output:08gru_1/while/gru_cell_1/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2)
'gru_1/while/gru_cell_1/strided_slice_11?
 gru_1/while/gru_cell_1/BiasAdd_5BiasAdd)gru_1/while/gru_cell_1/MatMul_5:product:00gru_1/while/gru_cell_1/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2"
 gru_1/while/gru_cell_1/BiasAdd_5?
gru_1/while/gru_cell_1/mul_3Mul$gru_1/while/gru_cell_1/Sigmoid_1:y:0)gru_1/while/gru_cell_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul_3?
gru_1/while/gru_cell_1/add_2AddV2)gru_1/while/gru_cell_1/BiasAdd_2:output:0 gru_1/while/gru_cell_1/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/add_2?
gru_1/while/gru_cell_1/TanhTanh gru_1/while/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/Tanh?
gru_1/while/gru_cell_1/mul_4Mul"gru_1/while/gru_cell_1/Sigmoid:y:0gru_1_while_placeholder_2*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul_4?
gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_1/while/gru_cell_1/sub/x?
gru_1/while/gru_cell_1/subSub%gru_1/while/gru_cell_1/sub/x:output:0"gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/sub?
gru_1/while/gru_cell_1/mul_5Mulgru_1/while/gru_cell_1/sub:z:0gru_1/while/gru_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/mul_5?
gru_1/while/gru_cell_1/add_3AddV2 gru_1/while/gru_cell_1/mul_4:z:0 gru_1/while/gru_cell_1/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_1/while/gru_cell_1/add_3?
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_1/while/TensorArrayV2Write/TensorListSetItemh
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/while/add/y?
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_1/while/addl
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/while/add_1/y?
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_1/while/add_1?
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_1/while/Identity?
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_1/while/Identity_1?
gru_1/while/Identity_2Identitygru_1/while/add:z:0&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_1/while/Identity_2?
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_1/while/Identity_3?
gru_1/while/Identity_4Identity gru_1/while/gru_cell_1/add_3:z:0&^gru_1/while/gru_cell_1/ReadVariableOp(^gru_1/while/gru_cell_1/ReadVariableOp_1(^gru_1/while/gru_cell_1/ReadVariableOp_2(^gru_1/while/gru_cell_1/ReadVariableOp_3(^gru_1/while/gru_cell_1/ReadVariableOp_4(^gru_1/while/gru_cell_1/ReadVariableOp_5(^gru_1/while/gru_cell_1/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
gru_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"f
0gru_1_while_gru_cell_1_readvariableop_1_resource2gru_1_while_gru_cell_1_readvariableop_1_resource_0"f
0gru_1_while_gru_cell_1_readvariableop_4_resource2gru_1_while_gru_cell_1_readvariableop_4_resource_0"b
.gru_1_while_gru_cell_1_readvariableop_resource0gru_1_while_gru_cell_1_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"?
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2N
%gru_1/while/gru_cell_1/ReadVariableOp%gru_1/while/gru_cell_1/ReadVariableOp2R
'gru_1/while/gru_cell_1/ReadVariableOp_1'gru_1/while/gru_cell_1/ReadVariableOp_12R
'gru_1/while/gru_cell_1/ReadVariableOp_2'gru_1/while/gru_cell_1/ReadVariableOp_22R
'gru_1/while/gru_cell_1/ReadVariableOp_3'gru_1/while/gru_cell_1/ReadVariableOp_32R
'gru_1/while/gru_cell_1/ReadVariableOp_4'gru_1/while/gru_cell_1/ReadVariableOp_42R
'gru_1/while/gru_cell_1/ReadVariableOp_5'gru_1/while/gru_cell_1/ReadVariableOp_52R
'gru_1/while/gru_cell_1/ReadVariableOp_6'gru_1/while/gru_cell_1/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_34
serving_default_input_3:0?????????	
;
input_40
serving_default_input_4:0?????????;
dense_70
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?n
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?j
_tf_keras_network?j{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}, "name": "gru_1", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_1", "inbound_nodes": [[["gru_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["layer_normalization_1", 0, 0, {}], ["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 11]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 14, 9]}, {"class_name": "TensorShape", "items": [null, 11]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}, "name": "gru_1", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_1", "inbound_nodes": [[["gru_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["layer_normalization_1", 0, 0, {}], ["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["dense_7", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 5.000000328436727e-06, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"class_name": "GRU", "name": "gru_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 9]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 9]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
?
axis
	gamma
beta
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 11}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11]}}
?
#trainable_variables
$	variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 200]}, {"class_name": "TensorShape", "items": [null, 200]}]}
?

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400]}}
?

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?

3kernel
4bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
9iter

:beta_1

;beta_2
	<decay
=learning_rate
>momentum_cachem?m?m?m?'m?(m?-m?.m?3m?4m??m?@m?Am?v?v?v?v?'v?(v?-v?.v?3v?4v??v?@v?Av?"
	optimizer
~
?0
@1
A2
3
4
5
6
'7
(8
-9
.10
311
412"
trackable_list_wrapper
~
?0
@1
A2
3
4
5
6
'7
(8
-9
.10
311
412"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Blayer_metrics
trainable_variables
	variables
Cnon_trainable_variables
Dlayer_regularization_losses

Elayers
Fmetrics
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

?kernel
@recurrent_kernel
Abias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GRUCell", "name": "gru_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}}
 "
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Kstates
Llayer_metrics
trainable_variables
Mnon_trainable_variables
Nlayer_regularization_losses
	variables

Olayers
Pmetrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(?2layer_normalization_1/gamma
):'?2layer_normalization_1/beta
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qlayer_metrics
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
	variables

Tlayers
Umetrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_4/kernel
:?2dense_4/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vlayer_metrics
trainable_variables
Wlayer_regularization_losses
Xnon_trainable_variables
 	variables

Ylayers
Zmetrics
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
[layer_metrics
#trainable_variables
\layer_regularization_losses
]non_trainable_variables
$	variables

^layers
_metrics
%regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_5/kernel
:?2dense_5/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
`layer_metrics
)trainable_variables
alayer_regularization_losses
bnon_trainable_variables
*	variables

clayers
dmetrics
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_6/kernel
:?2dense_6/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
elayer_metrics
/trainable_variables
flayer_regularization_losses
gnon_trainable_variables
0	variables

hlayers
imetrics
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_7/kernel
:2dense_7/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
jlayer_metrics
5trainable_variables
klayer_regularization_losses
lnon_trainable_variables
6	variables

mlayers
nmetrics
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
*:(		?2gru_1/gru_cell_1/kernel
5:3
??2!gru_1/gru_cell_1/recurrent_kernel
(:&	?2gru_1/gru_cell_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
qlayer_metrics
Gtrainable_variables
rlayer_regularization_losses
snon_trainable_variables
H	variables

tlayers
umetrics
Iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	vtotal
	wcount
x	variables
y	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?"
ztrue_positives
{true_negatives
|false_positives
}false_negatives
~	variables
	keras_api"?!
_tf_keras_metric?!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
v0
w1"
trackable_list_wrapper
-
x	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
<
z0
{1
|2
}3"
trackable_list_wrapper
-
~	variables"
_generic_user_object
0:.?2#Nadam/layer_normalization_1/gamma/m
/:-?2"Nadam/layer_normalization_1/beta/m
':%	?2Nadam/dense_4/kernel/m
!:?2Nadam/dense_4/bias/m
(:&
??2Nadam/dense_5/kernel/m
!:?2Nadam/dense_5/bias/m
(:&
??2Nadam/dense_6/kernel/m
!:?2Nadam/dense_6/bias/m
':%	?2Nadam/dense_7/kernel/m
 :2Nadam/dense_7/bias/m
0:.		?2Nadam/gru_1/gru_cell_1/kernel/m
;:9
??2)Nadam/gru_1/gru_cell_1/recurrent_kernel/m
.:,	?2Nadam/gru_1/gru_cell_1/bias/m
0:.?2#Nadam/layer_normalization_1/gamma/v
/:-?2"Nadam/layer_normalization_1/beta/v
':%	?2Nadam/dense_4/kernel/v
!:?2Nadam/dense_4/bias/v
(:&
??2Nadam/dense_5/kernel/v
!:?2Nadam/dense_5/bias/v
(:&
??2Nadam/dense_6/kernel/v
!:?2Nadam/dense_6/bias/v
':%	?2Nadam/dense_7/kernel/v
 :2Nadam/dense_7/bias/v
0:.		?2Nadam/gru_1/gru_cell_1/kernel/v
;:9
??2)Nadam/gru_1/gru_cell_1/recurrent_kernel/v
.:,	?2Nadam/gru_1/gru_cell_1/bias/v
?2?
(__inference_model_1_layer_call_fn_223297
(__inference_model_1_layer_call_fn_224201
(__inference_model_1_layer_call_fn_224169
(__inference_model_1_layer_call_fn_223367?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_221683?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *R?O
M?J
%?"
input_3?????????	
!?
input_4?????????
?2?
C__inference_model_1_layer_call_and_return_conditional_losses_223188
C__inference_model_1_layer_call_and_return_conditional_losses_224137
C__inference_model_1_layer_call_and_return_conditional_losses_223226
C__inference_model_1_layer_call_and_return_conditional_losses_223797?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_gru_1_layer_call_fn_225425
&__inference_gru_1_layer_call_fn_224813
&__inference_gru_1_layer_call_fn_225414
&__inference_gru_1_layer_call_fn_224802?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_gru_1_layer_call_and_return_conditional_losses_224791
A__inference_gru_1_layer_call_and_return_conditional_losses_225132
A__inference_gru_1_layer_call_and_return_conditional_losses_224520
A__inference_gru_1_layer_call_and_return_conditional_losses_225403?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_layer_normalization_1_layer_call_fn_225476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_225467?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_4_layer_call_fn_225496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_4_layer_call_and_return_conditional_losses_225487?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_concatenate_1_layer_call_fn_225509?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_concatenate_1_layer_call_and_return_conditional_losses_225503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_5_layer_call_fn_225529?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_5_layer_call_and_return_conditional_losses_225520?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_6_layer_call_fn_225549?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_6_layer_call_and_return_conditional_losses_225540?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_7_layer_call_fn_225569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_7_layer_call_and_return_conditional_losses_225560?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_223409input_3input_4"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_gru_cell_1_layer_call_fn_225799
+__inference_gru_cell_1_layer_call_fn_225813?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_225689
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_225785?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
!__inference__wrapped_model_221683?A?@'(-.34\?Y
R?O
M?J
%?"
input_3?????????	
!?
input_4?????????
? "1?.
,
dense_7!?
dense_7??????????
I__inference_concatenate_1_layer_call_and_return_conditional_losses_225503?\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
.__inference_concatenate_1_layer_call_fn_225509y\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "????????????
C__inference_dense_4_layer_call_and_return_conditional_losses_225487]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? |
(__inference_dense_4_layer_call_fn_225496P/?,
%?"
 ?
inputs?????????
? "????????????
C__inference_dense_5_layer_call_and_return_conditional_losses_225520^'(0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_5_layer_call_fn_225529Q'(0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_6_layer_call_and_return_conditional_losses_225540^-.0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_6_layer_call_fn_225549Q-.0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_7_layer_call_and_return_conditional_losses_225560]340?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_7_layer_call_fn_225569P340?-
&?#
!?
inputs??????????
? "???????????
A__inference_gru_1_layer_call_and_return_conditional_losses_224520~A?@O?L
E?B
4?1
/?,
inputs/0??????????????????	

 
p

 
? "&?#
?
0??????????
? ?
A__inference_gru_1_layer_call_and_return_conditional_losses_224791~A?@O?L
E?B
4?1
/?,
inputs/0??????????????????	

 
p 

 
? "&?#
?
0??????????
? ?
A__inference_gru_1_layer_call_and_return_conditional_losses_225132nA?@??<
5?2
$?!
inputs?????????	

 
p

 
? "&?#
?
0??????????
? ?
A__inference_gru_1_layer_call_and_return_conditional_losses_225403nA?@??<
5?2
$?!
inputs?????????	

 
p 

 
? "&?#
?
0??????????
? ?
&__inference_gru_1_layer_call_fn_224802qA?@O?L
E?B
4?1
/?,
inputs/0??????????????????	

 
p

 
? "????????????
&__inference_gru_1_layer_call_fn_224813qA?@O?L
E?B
4?1
/?,
inputs/0??????????????????	

 
p 

 
? "????????????
&__inference_gru_1_layer_call_fn_225414aA?@??<
5?2
$?!
inputs?????????	

 
p

 
? "????????????
&__inference_gru_1_layer_call_fn_225425aA?@??<
5?2
$?!
inputs?????????	

 
p 

 
? "????????????
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_225689?A?@]?Z
S?P
 ?
inputs?????????	
(?%
#? 
states/0??????????
p
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_225785?A?@]?Z
S?P
 ?
inputs?????????	
(?%
#? 
states/0??????????
p 
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
+__inference_gru_cell_1_layer_call_fn_225799?A?@]?Z
S?P
 ?
inputs?????????	
(?%
#? 
states/0??????????
p
? "F?C
?
0??????????
#? 
?
1/0???????????
+__inference_gru_cell_1_layer_call_fn_225813?A?@]?Z
S?P
 ?
inputs?????????	
(?%
#? 
states/0??????????
p 
? "F?C
?
0??????????
#? 
?
1/0???????????
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_225467^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
6__inference_layer_normalization_1_layer_call_fn_225476Q0?-
&?#
!?
inputs??????????
? "????????????
C__inference_model_1_layer_call_and_return_conditional_losses_223188?A?@'(-.34d?a
Z?W
M?J
%?"
input_3?????????	
!?
input_4?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_223226?A?@'(-.34d?a
Z?W
M?J
%?"
input_3?????????	
!?
input_4?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_223797?A?@'(-.34f?c
\?Y
O?L
&?#
inputs/0?????????	
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_224137?A?@'(-.34f?c
\?Y
O?L
&?#
inputs/0?????????	
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
(__inference_model_1_layer_call_fn_223297?A?@'(-.34d?a
Z?W
M?J
%?"
input_3?????????	
!?
input_4?????????
p

 
? "???????????
(__inference_model_1_layer_call_fn_223367?A?@'(-.34d?a
Z?W
M?J
%?"
input_3?????????	
!?
input_4?????????
p 

 
? "???????????
(__inference_model_1_layer_call_fn_224169?A?@'(-.34f?c
\?Y
O?L
&?#
inputs/0?????????	
"?
inputs/1?????????
p

 
? "???????????
(__inference_model_1_layer_call_fn_224201?A?@'(-.34f?c
\?Y
O?L
&?#
inputs/0?????????	
"?
inputs/1?????????
p 

 
? "???????????
$__inference_signature_wrapper_223409?A?@'(-.34m?j
? 
c?`
0
input_3%?"
input_3?????????	
,
input_4!?
input_4?????????"1?.
,
dense_7!?
dense_7?????????