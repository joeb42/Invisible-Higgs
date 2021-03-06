??+
??
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
?"serve*2.4.12unknown8??(
?
layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namelayer_normalization_2/gamma
?
/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes	
:?*
dtype0
?
layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelayer_normalization_2/beta
?
.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes	
:?*
dtype0
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes
:	?*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
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
gru_2/gru_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*(
shared_namegru_2/gru_cell_2/kernel
?
+gru_2/gru_cell_2/kernel/Read/ReadVariableOpReadVariableOpgru_2/gru_cell_2/kernel*
_output_shapes
:		?*
dtype0
?
!gru_2/gru_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!gru_2/gru_cell_2/recurrent_kernel
?
5gru_2/gru_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_2/gru_cell_2/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
gru_2/gru_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_namegru_2/gru_cell_2/bias
?
)gru_2/gru_cell_2/bias/Read/ReadVariableOpReadVariableOpgru_2/gru_cell_2/bias*
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
#Nadam/layer_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Nadam/layer_normalization_2/gamma/m
?
7Nadam/layer_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp#Nadam/layer_normalization_2/gamma/m*
_output_shapes	
:?*
dtype0
?
"Nadam/layer_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Nadam/layer_normalization_2/beta/m
?
6Nadam/layer_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp"Nadam/layer_normalization_2/beta/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameNadam/dense_12/kernel/m
?
+Nadam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_12/kernel/m*
_output_shapes
:	?*
dtype0
?
Nadam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_12/bias/m
{
)Nadam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_12/bias/m*
_output_shapes
:*
dtype0
?
Nadam/gru_2/gru_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*0
shared_name!Nadam/gru_2/gru_cell_2/kernel/m
?
3Nadam/gru_2/gru_cell_2/kernel/m/Read/ReadVariableOpReadVariableOpNadam/gru_2/gru_cell_2/kernel/m*
_output_shapes
:		?*
dtype0
?
)Nadam/gru_2/gru_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*:
shared_name+)Nadam/gru_2/gru_cell_2/recurrent_kernel/m
?
=Nadam/gru_2/gru_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp)Nadam/gru_2/gru_cell_2/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/gru_2/gru_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_nameNadam/gru_2/gru_cell_2/bias/m
?
1Nadam/gru_2/gru_cell_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/gru_2/gru_cell_2/bias/m*
_output_shapes
:	?*
dtype0
?
#Nadam/layer_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Nadam/layer_normalization_2/gamma/v
?
7Nadam/layer_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp#Nadam/layer_normalization_2/gamma/v*
_output_shapes	
:?*
dtype0
?
"Nadam/layer_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Nadam/layer_normalization_2/beta/v
?
6Nadam/layer_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp"Nadam/layer_normalization_2/beta/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameNadam/dense_12/kernel/v
?
+Nadam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_12/kernel/v*
_output_shapes
:	?*
dtype0
?
Nadam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_12/bias/v
{
)Nadam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_12/bias/v*
_output_shapes
:*
dtype0
?
Nadam/gru_2/gru_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*0
shared_name!Nadam/gru_2/gru_cell_2/kernel/v
?
3Nadam/gru_2/gru_cell_2/kernel/v/Read/ReadVariableOpReadVariableOpNadam/gru_2/gru_cell_2/kernel/v*
_output_shapes
:		?*
dtype0
?
)Nadam/gru_2/gru_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*:
shared_name+)Nadam/gru_2/gru_cell_2/recurrent_kernel/v
?
=Nadam/gru_2/gru_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp)Nadam/gru_2/gru_cell_2/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/gru_2/gru_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_nameNadam/gru_2/gru_cell_2/bias/v
?
1Nadam/gru_2/gru_cell_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/gru_2/gru_cell_2/bias/v*
_output_shapes
:	?*
dtype0

NoOpNoOp
?/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?.
value?.B?. B?.
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
 
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
q
axis
	gamma
beta
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?
iter

beta_1

 beta_2
	!decay
"learning_rate
#momentum_cachemQmRmSmT$mU%mV&mWvXvYvZv[$v\%v]&v^
 
1
$0
%1
&2
3
4
5
6
1
$0
%1
&2
3
4
5
6
?
regularization_losses
'layer_regularization_losses
	variables
(metrics
trainable_variables
)layer_metrics
*non_trainable_variables

+layers
 
~

$kernel
%recurrent_kernel
&bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
 
 

$0
%1
&2

$0
%1
&2
?
regularization_losses
0layer_regularization_losses
	variables
1metrics
trainable_variables

2states
3layer_metrics
4non_trainable_variables

5layers
 
fd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
6layer_regularization_losses
regularization_losses
	variables
7metrics
trainable_variables
8layer_metrics
9non_trainable_variables

:layers
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
;layer_regularization_losses
regularization_losses
	variables
<metrics
trainable_variables
=layer_metrics
>non_trainable_variables

?layers
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
SQ
VARIABLE_VALUEgru_2/gru_cell_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!gru_2/gru_cell_2/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEgru_2/gru_cell_2/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1
 
 

0
1
2
3
 

$0
%1
&2

$0
%1
&2
?
Blayer_regularization_losses
,regularization_losses
-	variables
Cmetrics
.trainable_variables
Dlayer_metrics
Enon_trainable_variables

Flayers
 
 
 
 
 

0
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
	Gtotal
	Hcount
I	variables
J	keras_api
p
Ktrue_positives
Ltrue_negatives
Mfalse_positives
Nfalse_negatives
O	variables
P	keras_api
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
G0
H1

I	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
M2
N3

O	variables
??
VARIABLE_VALUE#Nadam/layer_normalization_2/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Nadam/layer_normalization_2/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_12/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_12/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUENadam/gru_2/gru_cell_2/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Nadam/gru_2/gru_cell_2/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUENadam/gru_2/gru_cell_2/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Nadam/layer_normalization_2/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Nadam/layer_normalization_2/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_12/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_12/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUENadam/gru_2/gru_cell_2/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Nadam/gru_2/gru_cell_2/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUENadam/gru_2/gru_cell_2/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_5Placeholder*+
_output_shapes
:?????????	*
dtype0* 
shape:?????????	
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5gru_2/gru_cell_2/biasgru_2/gru_cell_2/kernel!gru_2/gru_cell_2/recurrent_kernellayer_normalization_2/gammalayer_normalization_2/betadense_12/kerneldense_12/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*2
config_proto" 

CPU

GPU2*3,4J 8? *-
f(R&
$__inference_signature_wrapper_565068
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOp+gru_2/gru_cell_2/kernel/Read/ReadVariableOp5gru_2/gru_cell_2/recurrent_kernel/Read/ReadVariableOp)gru_2/gru_cell_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp7Nadam/layer_normalization_2/gamma/m/Read/ReadVariableOp6Nadam/layer_normalization_2/beta/m/Read/ReadVariableOp+Nadam/dense_12/kernel/m/Read/ReadVariableOp)Nadam/dense_12/bias/m/Read/ReadVariableOp3Nadam/gru_2/gru_cell_2/kernel/m/Read/ReadVariableOp=Nadam/gru_2/gru_cell_2/recurrent_kernel/m/Read/ReadVariableOp1Nadam/gru_2/gru_cell_2/bias/m/Read/ReadVariableOp7Nadam/layer_normalization_2/gamma/v/Read/ReadVariableOp6Nadam/layer_normalization_2/beta/v/Read/ReadVariableOp+Nadam/dense_12/kernel/v/Read/ReadVariableOp)Nadam/dense_12/bias/v/Read/ReadVariableOp3Nadam/gru_2/gru_cell_2/kernel/v/Read/ReadVariableOp=Nadam/gru_2/gru_cell_2/recurrent_kernel/v/Read/ReadVariableOp1Nadam/gru_2/gru_cell_2/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*3,4J 8? *(
f#R!
__inference__traced_save_567447
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization_2/gammalayer_normalization_2/betadense_12/kerneldense_12/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachegru_2/gru_cell_2/kernel!gru_2/gru_cell_2/recurrent_kernelgru_2/gru_cell_2/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negatives#Nadam/layer_normalization_2/gamma/m"Nadam/layer_normalization_2/beta/mNadam/dense_12/kernel/mNadam/dense_12/bias/mNadam/gru_2/gru_cell_2/kernel/m)Nadam/gru_2/gru_cell_2/recurrent_kernel/mNadam/gru_2/gru_cell_2/bias/m#Nadam/layer_normalization_2/gamma/v"Nadam/layer_normalization_2/beta/vNadam/dense_12/kernel/vNadam/dense_12/bias/vNadam/gru_2/gru_cell_2/kernel/v)Nadam/gru_2/gru_cell_2/recurrent_kernel/vNadam/gru_2/gru_cell_2/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*3,4J 8? *+
f&R$
"__inference__traced_restore_567556??'
?
?
6__inference_layer_normalization_2_layer_call_fn_567061

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
*2
config_proto" 

CPU

GPU2*3,4J 8? *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_5648932
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
?
$__inference_signature_wrapper_565068
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*2
config_proto" 

CPU

GPU2*3,4J 8? **
f%R#
!__inference__wrapped_model_5635302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
?

?
model_4_gru_2_while_cond_5633388
4model_4_gru_2_while_model_4_gru_2_while_loop_counter>
:model_4_gru_2_while_model_4_gru_2_while_maximum_iterations#
model_4_gru_2_while_placeholder%
!model_4_gru_2_while_placeholder_1%
!model_4_gru_2_while_placeholder_2:
6model_4_gru_2_while_less_model_4_gru_2_strided_slice_1P
Lmodel_4_gru_2_while_model_4_gru_2_while_cond_563338___redundant_placeholder0P
Lmodel_4_gru_2_while_model_4_gru_2_while_cond_563338___redundant_placeholder1P
Lmodel_4_gru_2_while_model_4_gru_2_while_cond_563338___redundant_placeholder2P
Lmodel_4_gru_2_while_model_4_gru_2_while_cond_563338___redundant_placeholder3 
model_4_gru_2_while_identity
?
model_4/gru_2/while/LessLessmodel_4_gru_2_while_placeholder6model_4_gru_2_while_less_model_4_gru_2_strided_slice_1*
T0*
_output_shapes
: 2
model_4/gru_2/while/Less?
model_4/gru_2/while/IdentityIdentitymodel_4/gru_2/while/Less:z:0*
T0
*
_output_shapes
: 2
model_4/gru_2/while/Identity"E
model_4_gru_2_while_identity%model_4/gru_2/while/Identity:output:0*A
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
A__inference_gru_2_layer_call_and_return_conditional_losses_564821

inputs&
"gru_cell_2_readvariableop_resource(
$gru_cell_2_readvariableop_1_resource(
$gru_cell_2_readvariableop_4_resource
identity??gru_cell_2/ReadVariableOp?gru_cell_2/ReadVariableOp_1?gru_cell_2/ReadVariableOp_2?gru_cell_2/ReadVariableOp_3?gru_cell_2/ReadVariableOp_4?gru_cell_2/ReadVariableOp_5?gru_cell_2/ReadVariableOp_6?whileD
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
gru_cell_2/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/ones_like/Const?
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/ones_like?
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_2/ReadVariableOp?
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_2/unstack?
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_1?
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack?
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice/stack_1?
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2?
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice?
gru_cell_2/MatMulMatMulstrided_slice_2:output:0!gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul?
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_2?
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_1/stack?
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_1/stack_1?
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2?
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_1?
gru_cell_2/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_1?
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_3?
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_2/strided_slice_2/stack?
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1?
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2?
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_2?
gru_cell_2/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_2?
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack?
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_3/stack_1?
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2?
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_3?
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd?
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_4/stack?
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_4/stack_1?
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2?
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_4?
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_1?
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_5/stack?
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1?
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2?
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_5?
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_2?
gru_cell_2/mulMulzeros:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul?
gru_cell_2/mul_1Mulzeros:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_1?
gru_cell_2/mul_2Mulzeros:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_2?
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_4?
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack?
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_2/strided_slice_6/stack_1?
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2?
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_6?
gru_cell_2/MatMul_3MatMulgru_cell_2/mul:z:0#gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_3?
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_5?
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_7/stack?
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_7/stack_1?
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2?
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_7?
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_4?
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack?
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_8/stack_1?
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2?
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_8?
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_3?
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_9/stack?
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_9/stack_1?
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2?
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_9?
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_4?
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/addz
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid?
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_1?
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid_1?
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_6?
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_2/strided_slice_10/stack?
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1?
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2?
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_10?
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_2:z:0$gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_5?
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_2/strided_slice_11/stack?
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1?
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2?
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_11?
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_5?
gru_cell_2/mul_3Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_3?
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_2s
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Tanh?
gru_cell_2/mul_4Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_4i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/sub/x?
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/sub?
gru_cell_2/mul_5Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_5?
gru_cell_2/add_3AddV2gru_cell_2/mul_4:z:0gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
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
while_body_564675*
condR
while_cond_564674*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
A__inference_gru_2_layer_call_and_return_conditional_losses_566717

inputs&
"gru_cell_2_readvariableop_resource(
$gru_cell_2_readvariableop_1_resource(
$gru_cell_2_readvariableop_4_resource
identity??gru_cell_2/ReadVariableOp?gru_cell_2/ReadVariableOp_1?gru_cell_2/ReadVariableOp_2?gru_cell_2/ReadVariableOp_3?gru_cell_2/ReadVariableOp_4?gru_cell_2/ReadVariableOp_5?gru_cell_2/ReadVariableOp_6?whileD
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
gru_cell_2/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/ones_like/Const?
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/ones_likey
gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout/Const?
gru_cell_2/dropout/MulMulgru_cell_2/ones_like:output:0!gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout/Mul?
gru_cell_2/dropout/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout/Shape?
/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???21
/gru_cell_2/dropout/random_uniform/RandomUniform?
!gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell_2/dropout/GreaterEqual/y?
gru_cell_2/dropout/GreaterEqualGreaterEqual8gru_cell_2/dropout/random_uniform/RandomUniform:output:0*gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell_2/dropout/GreaterEqual?
gru_cell_2/dropout/CastCast#gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_2/dropout/Cast?
gru_cell_2/dropout/Mul_1Mulgru_cell_2/dropout/Mul:z:0gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout/Mul_1}
gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_1/Const?
gru_cell_2/dropout_1/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_1/Mul?
gru_cell_2/dropout_1/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_1/Shape?
1gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???23
1gru_cell_2/dropout_1/random_uniform/RandomUniform?
#gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_2/dropout_1/GreaterEqual/y?
!gru_cell_2/dropout_1/GreaterEqualGreaterEqual:gru_cell_2/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_2/dropout_1/GreaterEqual?
gru_cell_2/dropout_1/CastCast%gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_2/dropout_1/Cast?
gru_cell_2/dropout_1/Mul_1Mulgru_cell_2/dropout_1/Mul:z:0gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_1/Mul_1}
gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_2/Const?
gru_cell_2/dropout_2/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_2/Mul?
gru_cell_2/dropout_2/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_2/Shape?
1gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??M23
1gru_cell_2/dropout_2/random_uniform/RandomUniform?
#gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_2/dropout_2/GreaterEqual/y?
!gru_cell_2/dropout_2/GreaterEqualGreaterEqual:gru_cell_2/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_2/dropout_2/GreaterEqual?
gru_cell_2/dropout_2/CastCast%gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_2/dropout_2/Cast?
gru_cell_2/dropout_2/Mul_1Mulgru_cell_2/dropout_2/Mul:z:0gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_2/Mul_1?
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_2/ReadVariableOp?
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_2/unstack?
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_1?
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack?
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice/stack_1?
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2?
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice?
gru_cell_2/MatMulMatMulstrided_slice_2:output:0!gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul?
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_2?
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_1/stack?
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_1/stack_1?
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2?
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_1?
gru_cell_2/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_1?
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_3?
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_2/strided_slice_2/stack?
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1?
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2?
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_2?
gru_cell_2/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_2?
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack?
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_3/stack_1?
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2?
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_3?
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd?
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_4/stack?
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_4/stack_1?
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2?
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_4?
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_1?
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_5/stack?
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1?
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2?
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_5?
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_2?
gru_cell_2/mulMulzeros:output:0gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul?
gru_cell_2/mul_1Mulzeros:output:0gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_1?
gru_cell_2/mul_2Mulzeros:output:0gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_2?
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_4?
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack?
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_2/strided_slice_6/stack_1?
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2?
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_6?
gru_cell_2/MatMul_3MatMulgru_cell_2/mul:z:0#gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_3?
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_5?
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_7/stack?
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_7/stack_1?
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2?
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_7?
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_4?
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack?
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_8/stack_1?
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2?
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_8?
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_3?
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_9/stack?
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_9/stack_1?
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2?
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_9?
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_4?
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/addz
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid?
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_1?
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid_1?
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_6?
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_2/strided_slice_10/stack?
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1?
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2?
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_10?
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_2:z:0$gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_5?
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_2/strided_slice_11/stack?
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1?
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2?
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_11?
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_5?
gru_cell_2/mul_3Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_3?
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_2s
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Tanh?
gru_cell_2/mul_4Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_4i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/sub/x?
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/sub?
gru_cell_2/mul_5Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_5?
gru_cell_2/add_3AddV2gru_cell_2/mul_4:z:0gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
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
while_body_566547*
condR
while_cond_566546*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?<
?
A__inference_gru_2_layer_call_and_return_conditional_losses_564219

inputs
gru_cell_2_564143
gru_cell_2_564145
gru_cell_2_564147
identity??"gru_cell_2/StatefulPartitionedCall?whileD
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
"gru_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_2_564143gru_cell_2_564145gru_cell_2_564147*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_5637782$
"gru_cell_2/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_2_564143gru_cell_2_564145gru_cell_2_564147*
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
while_body_564155*
condR
while_cond_564154*9
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
IdentityIdentitystrided_slice_3:output:0#^gru_cell_2/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::2H
"gru_cell_2/StatefulPartitionedCall"gru_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????	
 
_user_specified_nameinputs
?
?
while_cond_566229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_566229___redundant_placeholder04
0while_while_cond_566229___redundant_placeholder14
0while_while_cond_566229___redundant_placeholder24
0while_while_cond_566229___redundant_placeholder3
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
?	
gru_2_while_body_565557(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_04
0gru_2_while_gru_cell_2_readvariableop_resource_06
2gru_2_while_gru_cell_2_readvariableop_1_resource_06
2gru_2_while_gru_cell_2_readvariableop_4_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor2
.gru_2_while_gru_cell_2_readvariableop_resource4
0gru_2_while_gru_cell_2_readvariableop_1_resource4
0gru_2_while_gru_cell_2_readvariableop_4_resource??%gru_2/while/gru_cell_2/ReadVariableOp?'gru_2/while/gru_cell_2/ReadVariableOp_1?'gru_2/while/gru_cell_2/ReadVariableOp_2?'gru_2/while/gru_cell_2/ReadVariableOp_3?'gru_2/while/gru_cell_2/ReadVariableOp_4?'gru_2/while/gru_cell_2/ReadVariableOp_5?'gru_2/while/gru_cell_2/ReadVariableOp_6?
=gru_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2?
=gru_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
/gru_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0gru_2_while_placeholderFgru_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype021
/gru_2/while/TensorArrayV2Read/TensorListGetItem?
&gru_2/while/gru_cell_2/ones_like/ShapeShapegru_2_while_placeholder_2*
T0*
_output_shapes
:2(
&gru_2/while/gru_cell_2/ones_like/Shape?
&gru_2/while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&gru_2/while/gru_cell_2/ones_like/Const?
 gru_2/while/gru_cell_2/ones_likeFill/gru_2/while/gru_cell_2/ones_like/Shape:output:0/gru_2/while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/ones_like?
%gru_2/while/gru_cell_2/ReadVariableOpReadVariableOp0gru_2_while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	?*
dtype02'
%gru_2/while/gru_cell_2/ReadVariableOp?
gru_2/while/gru_cell_2/unstackUnpack-gru_2/while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2 
gru_2/while/gru_cell_2/unstack?
'gru_2/while/gru_cell_2/ReadVariableOp_1ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_1?
*gru_2/while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*gru_2/while/gru_cell_2/strided_slice/stack?
,gru_2/while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_2/while/gru_cell_2/strided_slice/stack_1?
,gru_2/while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru_2/while/gru_cell_2/strided_slice/stack_2?
$gru_2/while/gru_cell_2/strided_sliceStridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_1:value:03gru_2/while/gru_cell_2/strided_slice/stack:output:05gru_2/while/gru_cell_2/strided_slice/stack_1:output:05gru_2/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2&
$gru_2/while/gru_cell_2/strided_slice?
gru_2/while/gru_cell_2/MatMulMatMul6gru_2/while/TensorArrayV2Read/TensorListGetItem:item:0-gru_2/while/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/MatMul?
'gru_2/while/gru_cell_2/ReadVariableOp_2ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_2?
,gru_2/while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_2/while/gru_cell_2/strided_slice_1/stack?
.gru_2/while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  20
.gru_2/while/gru_cell_2/strided_slice_1/stack_1?
.gru_2/while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_2/while/gru_cell_2/strided_slice_1/stack_2?
&gru_2/while/gru_cell_2/strided_slice_1StridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_2:value:05gru_2/while/gru_cell_2/strided_slice_1/stack:output:07gru_2/while/gru_cell_2/strided_slice_1/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&gru_2/while/gru_cell_2/strided_slice_1?
gru_2/while/gru_cell_2/MatMul_1MatMul6gru_2/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_2/while/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
gru_2/while/gru_cell_2/MatMul_1?
'gru_2/while/gru_cell_2/ReadVariableOp_3ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_3?
,gru_2/while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2.
,gru_2/while/gru_cell_2/strided_slice_2/stack?
.gru_2/while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.gru_2/while/gru_cell_2/strided_slice_2/stack_1?
.gru_2/while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_2/while/gru_cell_2/strided_slice_2/stack_2?
&gru_2/while/gru_cell_2/strided_slice_2StridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_3:value:05gru_2/while/gru_cell_2/strided_slice_2/stack:output:07gru_2/while/gru_cell_2/strided_slice_2/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&gru_2/while/gru_cell_2/strided_slice_2?
gru_2/while/gru_cell_2/MatMul_2MatMul6gru_2/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_2/while/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
gru_2/while/gru_cell_2/MatMul_2?
,gru_2/while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,gru_2/while/gru_cell_2/strided_slice_3/stack?
.gru_2/while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_2/while/gru_cell_2/strided_slice_3/stack_1?
.gru_2/while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_2/while/gru_cell_2/strided_slice_3/stack_2?
&gru_2/while/gru_cell_2/strided_slice_3StridedSlice'gru_2/while/gru_cell_2/unstack:output:05gru_2/while/gru_cell_2/strided_slice_3/stack:output:07gru_2/while/gru_cell_2/strided_slice_3/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2(
&gru_2/while/gru_cell_2/strided_slice_3?
gru_2/while/gru_cell_2/BiasAddBiasAdd'gru_2/while/gru_cell_2/MatMul:product:0/gru_2/while/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2 
gru_2/while/gru_cell_2/BiasAdd?
,gru_2/while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_2/while/gru_cell_2/strided_slice_4/stack?
.gru_2/while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_2/while/gru_cell_2/strided_slice_4/stack_1?
.gru_2/while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_2/while/gru_cell_2/strided_slice_4/stack_2?
&gru_2/while/gru_cell_2/strided_slice_4StridedSlice'gru_2/while/gru_cell_2/unstack:output:05gru_2/while/gru_cell_2/strided_slice_4/stack:output:07gru_2/while/gru_cell_2/strided_slice_4/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2(
&gru_2/while/gru_cell_2/strided_slice_4?
 gru_2/while/gru_cell_2/BiasAdd_1BiasAdd)gru_2/while/gru_cell_2/MatMul_1:product:0/gru_2/while/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/BiasAdd_1?
,gru_2/while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_2/while/gru_cell_2/strided_slice_5/stack?
.gru_2/while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.gru_2/while/gru_cell_2/strided_slice_5/stack_1?
.gru_2/while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_2/while/gru_cell_2/strided_slice_5/stack_2?
&gru_2/while/gru_cell_2/strided_slice_5StridedSlice'gru_2/while/gru_cell_2/unstack:output:05gru_2/while/gru_cell_2/strided_slice_5/stack:output:07gru_2/while/gru_cell_2/strided_slice_5/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2(
&gru_2/while/gru_cell_2/strided_slice_5?
 gru_2/while/gru_cell_2/BiasAdd_2BiasAdd)gru_2/while/gru_cell_2/MatMul_2:product:0/gru_2/while/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/BiasAdd_2?
gru_2/while/gru_cell_2/mulMulgru_2_while_placeholder_2)gru_2/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul?
gru_2/while/gru_cell_2/mul_1Mulgru_2_while_placeholder_2)gru_2/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul_1?
gru_2/while/gru_cell_2/mul_2Mulgru_2_while_placeholder_2)gru_2/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul_2?
'gru_2/while/gru_cell_2/ReadVariableOp_4ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_4?
,gru_2/while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,gru_2/while/gru_cell_2/strided_slice_6/stack?
.gru_2/while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   20
.gru_2/while/gru_cell_2/strided_slice_6/stack_1?
.gru_2/while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_2/while/gru_cell_2/strided_slice_6/stack_2?
&gru_2/while/gru_cell_2/strided_slice_6StridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_4:value:05gru_2/while/gru_cell_2/strided_slice_6/stack:output:07gru_2/while/gru_cell_2/strided_slice_6/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&gru_2/while/gru_cell_2/strided_slice_6?
gru_2/while/gru_cell_2/MatMul_3MatMulgru_2/while/gru_cell_2/mul:z:0/gru_2/while/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2!
gru_2/while/gru_cell_2/MatMul_3?
'gru_2/while/gru_cell_2/ReadVariableOp_5ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_5?
,gru_2/while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_2/while/gru_cell_2/strided_slice_7/stack?
.gru_2/while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  20
.gru_2/while/gru_cell_2/strided_slice_7/stack_1?
.gru_2/while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_2/while/gru_cell_2/strided_slice_7/stack_2?
&gru_2/while/gru_cell_2/strided_slice_7StridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_5:value:05gru_2/while/gru_cell_2/strided_slice_7/stack:output:07gru_2/while/gru_cell_2/strided_slice_7/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&gru_2/while/gru_cell_2/strided_slice_7?
gru_2/while/gru_cell_2/MatMul_4MatMul gru_2/while/gru_cell_2/mul_1:z:0/gru_2/while/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2!
gru_2/while/gru_cell_2/MatMul_4?
,gru_2/while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,gru_2/while/gru_cell_2/strided_slice_8/stack?
.gru_2/while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_2/while/gru_cell_2/strided_slice_8/stack_1?
.gru_2/while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_2/while/gru_cell_2/strided_slice_8/stack_2?
&gru_2/while/gru_cell_2/strided_slice_8StridedSlice'gru_2/while/gru_cell_2/unstack:output:15gru_2/while/gru_cell_2/strided_slice_8/stack:output:07gru_2/while/gru_cell_2/strided_slice_8/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2(
&gru_2/while/gru_cell_2/strided_slice_8?
 gru_2/while/gru_cell_2/BiasAdd_3BiasAdd)gru_2/while/gru_cell_2/MatMul_3:product:0/gru_2/while/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/BiasAdd_3?
,gru_2/while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_2/while/gru_cell_2/strided_slice_9/stack?
.gru_2/while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_2/while/gru_cell_2/strided_slice_9/stack_1?
.gru_2/while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_2/while/gru_cell_2/strided_slice_9/stack_2?
&gru_2/while/gru_cell_2/strided_slice_9StridedSlice'gru_2/while/gru_cell_2/unstack:output:15gru_2/while/gru_cell_2/strided_slice_9/stack:output:07gru_2/while/gru_cell_2/strided_slice_9/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2(
&gru_2/while/gru_cell_2/strided_slice_9?
 gru_2/while/gru_cell_2/BiasAdd_4BiasAdd)gru_2/while/gru_cell_2/MatMul_4:product:0/gru_2/while/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/BiasAdd_4?
gru_2/while/gru_cell_2/addAddV2'gru_2/while/gru_cell_2/BiasAdd:output:0)gru_2/while/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/add?
gru_2/while/gru_cell_2/SigmoidSigmoidgru_2/while/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2 
gru_2/while/gru_cell_2/Sigmoid?
gru_2/while/gru_cell_2/add_1AddV2)gru_2/while/gru_cell_2/BiasAdd_1:output:0)gru_2/while/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/add_1?
 gru_2/while/gru_cell_2/Sigmoid_1Sigmoid gru_2/while/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/Sigmoid_1?
'gru_2/while/gru_cell_2/ReadVariableOp_6ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_6?
-gru_2/while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2/
-gru_2/while/gru_cell_2/strided_slice_10/stack?
/gru_2/while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/gru_2/while/gru_cell_2/strided_slice_10/stack_1?
/gru_2/while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/gru_2/while/gru_cell_2/strided_slice_10/stack_2?
'gru_2/while/gru_cell_2/strided_slice_10StridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_6:value:06gru_2/while/gru_cell_2/strided_slice_10/stack:output:08gru_2/while/gru_cell_2/strided_slice_10/stack_1:output:08gru_2/while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2)
'gru_2/while/gru_cell_2/strided_slice_10?
gru_2/while/gru_cell_2/MatMul_5MatMul gru_2/while/gru_cell_2/mul_2:z:00gru_2/while/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2!
gru_2/while/gru_cell_2/MatMul_5?
-gru_2/while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2/
-gru_2/while/gru_cell_2/strided_slice_11/stack?
/gru_2/while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/gru_2/while/gru_cell_2/strided_slice_11/stack_1?
/gru_2/while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/gru_2/while/gru_cell_2/strided_slice_11/stack_2?
'gru_2/while/gru_cell_2/strided_slice_11StridedSlice'gru_2/while/gru_cell_2/unstack:output:16gru_2/while/gru_cell_2/strided_slice_11/stack:output:08gru_2/while/gru_cell_2/strided_slice_11/stack_1:output:08gru_2/while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2)
'gru_2/while/gru_cell_2/strided_slice_11?
 gru_2/while/gru_cell_2/BiasAdd_5BiasAdd)gru_2/while/gru_cell_2/MatMul_5:product:00gru_2/while/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/BiasAdd_5?
gru_2/while/gru_cell_2/mul_3Mul$gru_2/while/gru_cell_2/Sigmoid_1:y:0)gru_2/while/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul_3?
gru_2/while/gru_cell_2/add_2AddV2)gru_2/while/gru_cell_2/BiasAdd_2:output:0 gru_2/while/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/add_2?
gru_2/while/gru_cell_2/TanhTanh gru_2/while/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/Tanh?
gru_2/while/gru_cell_2/mul_4Mul"gru_2/while/gru_cell_2/Sigmoid:y:0gru_2_while_placeholder_2*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul_4?
gru_2/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_2/while/gru_cell_2/sub/x?
gru_2/while/gru_cell_2/subSub%gru_2/while/gru_cell_2/sub/x:output:0"gru_2/while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/sub?
gru_2/while/gru_cell_2/mul_5Mulgru_2/while/gru_cell_2/sub:z:0gru_2/while/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul_5?
gru_2/while/gru_cell_2/add_3AddV2 gru_2/while/gru_cell_2/mul_4:z:0 gru_2/while/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/add_3?
0gru_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder gru_2/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_2/while/TensorArrayV2Write/TensorListSetItemh
gru_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_2/while/add/y?
gru_2/while/addAddV2gru_2_while_placeholdergru_2/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_2/while/addl
gru_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_2/while/add_1/y?
gru_2/while/add_1AddV2$gru_2_while_gru_2_while_loop_countergru_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_2/while/add_1?
gru_2/while/IdentityIdentitygru_2/while/add_1:z:0&^gru_2/while/gru_cell_2/ReadVariableOp(^gru_2/while/gru_cell_2/ReadVariableOp_1(^gru_2/while/gru_cell_2/ReadVariableOp_2(^gru_2/while/gru_cell_2/ReadVariableOp_3(^gru_2/while/gru_cell_2/ReadVariableOp_4(^gru_2/while/gru_cell_2/ReadVariableOp_5(^gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_2/while/Identity?
gru_2/while/Identity_1Identity*gru_2_while_gru_2_while_maximum_iterations&^gru_2/while/gru_cell_2/ReadVariableOp(^gru_2/while/gru_cell_2/ReadVariableOp_1(^gru_2/while/gru_cell_2/ReadVariableOp_2(^gru_2/while/gru_cell_2/ReadVariableOp_3(^gru_2/while/gru_cell_2/ReadVariableOp_4(^gru_2/while/gru_cell_2/ReadVariableOp_5(^gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_2/while/Identity_1?
gru_2/while/Identity_2Identitygru_2/while/add:z:0&^gru_2/while/gru_cell_2/ReadVariableOp(^gru_2/while/gru_cell_2/ReadVariableOp_1(^gru_2/while/gru_cell_2/ReadVariableOp_2(^gru_2/while/gru_cell_2/ReadVariableOp_3(^gru_2/while/gru_cell_2/ReadVariableOp_4(^gru_2/while/gru_cell_2/ReadVariableOp_5(^gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_2/while/Identity_2?
gru_2/while/Identity_3Identity@gru_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^gru_2/while/gru_cell_2/ReadVariableOp(^gru_2/while/gru_cell_2/ReadVariableOp_1(^gru_2/while/gru_cell_2/ReadVariableOp_2(^gru_2/while/gru_cell_2/ReadVariableOp_3(^gru_2/while/gru_cell_2/ReadVariableOp_4(^gru_2/while/gru_cell_2/ReadVariableOp_5(^gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_2/while/Identity_3?
gru_2/while/Identity_4Identity gru_2/while/gru_cell_2/add_3:z:0&^gru_2/while/gru_cell_2/ReadVariableOp(^gru_2/while/gru_cell_2/ReadVariableOp_1(^gru_2/while/gru_cell_2/ReadVariableOp_2(^gru_2/while/gru_cell_2/ReadVariableOp_3(^gru_2/while/gru_cell_2/ReadVariableOp_4(^gru_2/while/gru_cell_2/ReadVariableOp_5(^gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
gru_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"f
0gru_2_while_gru_cell_2_readvariableop_1_resource2gru_2_while_gru_cell_2_readvariableop_1_resource_0"f
0gru_2_while_gru_cell_2_readvariableop_4_resource2gru_2_while_gru_cell_2_readvariableop_4_resource_0"b
.gru_2_while_gru_cell_2_readvariableop_resource0gru_2_while_gru_cell_2_readvariableop_resource_0"5
gru_2_while_identitygru_2/while/Identity:output:0"9
gru_2_while_identity_1gru_2/while/Identity_1:output:0"9
gru_2_while_identity_2gru_2/while/Identity_2:output:0"9
gru_2_while_identity_3gru_2/while/Identity_3:output:0"9
gru_2_while_identity_4gru_2/while/Identity_4:output:0"?
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2N
%gru_2/while/gru_cell_2/ReadVariableOp%gru_2/while/gru_cell_2/ReadVariableOp2R
'gru_2/while/gru_cell_2/ReadVariableOp_1'gru_2/while/gru_cell_2/ReadVariableOp_12R
'gru_2/while/gru_cell_2/ReadVariableOp_2'gru_2/while/gru_cell_2/ReadVariableOp_22R
'gru_2/while/gru_cell_2/ReadVariableOp_3'gru_2/while/gru_cell_2/ReadVariableOp_32R
'gru_2/while/gru_cell_2/ReadVariableOp_4'gru_2/while/gru_cell_2/ReadVariableOp_42R
'gru_2/while/gru_cell_2/ReadVariableOp_5'gru_2/while/gru_cell_2/ReadVariableOp_52R
'gru_2/while/gru_cell_2/ReadVariableOp_6'gru_2/while/gru_cell_2/ReadVariableOp_6: 
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
?
?
C__inference_model_4_layer_call_and_return_conditional_losses_564982

inputs
gru_2_564964
gru_2_564966
gru_2_564968 
layer_normalization_2_564971 
layer_normalization_2_564973
dense_12_564976
dense_12_564978
identity?? dense_12/StatefulPartitionedCall?gru_2/StatefulPartitionedCall?-layer_normalization_2/StatefulPartitionedCall?
gru_2/StatefulPartitionedCallStatefulPartitionedCallinputsgru_2_564964gru_2_564966gru_2_564968*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *J
fERC
A__inference_gru_2_layer_call_and_return_conditional_losses_5645502
gru_2/StatefulPartitionedCall?
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&gru_2/StatefulPartitionedCall:output:0layer_normalization_2_564971layer_normalization_2_564973*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_5648932/
-layer_normalization_2/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_12_564976dense_12_564978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5649202"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall^gru_2/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2>
gru_2/StatefulPartitionedCallgru_2/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
while_cond_566546
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_566546___redundant_placeholder04
0while_while_cond_566546___redundant_placeholder14
0while_while_cond_566546___redundant_placeholder24
0while_while_cond_566546___redundant_placeholder3
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
?h
?
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_563778

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
?"
?
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_564893

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
??
?
C__inference_model_4_layer_call_and_return_conditional_losses_565748

inputs,
(gru_2_gru_cell_2_readvariableop_resource.
*gru_2_gru_cell_2_readvariableop_1_resource.
*gru_2_gru_cell_2_readvariableop_4_resource7
3layer_normalization_2_mul_2_readvariableop_resource5
1layer_normalization_2_add_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?gru_2/gru_cell_2/ReadVariableOp?!gru_2/gru_cell_2/ReadVariableOp_1?!gru_2/gru_cell_2/ReadVariableOp_2?!gru_2/gru_cell_2/ReadVariableOp_3?!gru_2/gru_cell_2/ReadVariableOp_4?!gru_2/gru_cell_2/ReadVariableOp_5?!gru_2/gru_cell_2/ReadVariableOp_6?gru_2/while?(layer_normalization_2/add/ReadVariableOp?*layer_normalization_2/mul_2/ReadVariableOpP
gru_2/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_2/Shape?
gru_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_2/strided_slice/stack?
gru_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice/stack_1?
gru_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice/stack_2?
gru_2/strided_sliceStridedSlicegru_2/Shape:output:0"gru_2/strided_slice/stack:output:0$gru_2/strided_slice/stack_1:output:0$gru_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_2/strided_slicei
gru_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_2/zeros/mul/y?
gru_2/zeros/mulMulgru_2/strided_slice:output:0gru_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_2/zeros/mulk
gru_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_2/zeros/Less/y
gru_2/zeros/LessLessgru_2/zeros/mul:z:0gru_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_2/zeros/Lesso
gru_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_2/zeros/packed/1?
gru_2/zeros/packedPackgru_2/strided_slice:output:0gru_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_2/zeros/packedk
gru_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_2/zeros/Const?
gru_2/zerosFillgru_2/zeros/packed:output:0gru_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_2/zeros?
gru_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_2/transpose/perm?
gru_2/transpose	Transposeinputsgru_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????	2
gru_2/transposea
gru_2/Shape_1Shapegru_2/transpose:y:0*
T0*
_output_shapes
:2
gru_2/Shape_1?
gru_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_2/strided_slice_1/stack?
gru_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice_1/stack_1?
gru_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice_1/stack_2?
gru_2/strided_slice_1StridedSlicegru_2/Shape_1:output:0$gru_2/strided_slice_1/stack:output:0&gru_2/strided_slice_1/stack_1:output:0&gru_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_2/strided_slice_1?
!gru_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru_2/TensorArrayV2/element_shape?
gru_2/TensorArrayV2TensorListReserve*gru_2/TensorArrayV2/element_shape:output:0gru_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_2/TensorArrayV2?
;gru_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2=
;gru_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
-gru_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_2/transpose:y:0Dgru_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_2/TensorArrayUnstack/TensorListFromTensor?
gru_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_2/strided_slice_2/stack?
gru_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice_2/stack_1?
gru_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice_2/stack_2?
gru_2/strided_slice_2StridedSlicegru_2/transpose:y:0$gru_2/strided_slice_2/stack:output:0&gru_2/strided_slice_2/stack_1:output:0&gru_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
gru_2/strided_slice_2?
 gru_2/gru_cell_2/ones_like/ShapeShapegru_2/zeros:output:0*
T0*
_output_shapes
:2"
 gru_2/gru_cell_2/ones_like/Shape?
 gru_2/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 gru_2/gru_cell_2/ones_like/Const?
gru_2/gru_cell_2/ones_likeFill)gru_2/gru_cell_2/ones_like/Shape:output:0)gru_2/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/ones_like?
gru_2/gru_cell_2/ReadVariableOpReadVariableOp(gru_2_gru_cell_2_readvariableop_resource*
_output_shapes
:	?*
dtype02!
gru_2/gru_cell_2/ReadVariableOp?
gru_2/gru_cell_2/unstackUnpack'gru_2/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_2/gru_cell_2/unstack?
!gru_2/gru_cell_2/ReadVariableOp_1ReadVariableOp*gru_2_gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_1?
$gru_2/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$gru_2/gru_cell_2/strided_slice/stack?
&gru_2/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_2/gru_cell_2/strided_slice/stack_1?
&gru_2/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru_2/gru_cell_2/strided_slice/stack_2?
gru_2/gru_cell_2/strided_sliceStridedSlice)gru_2/gru_cell_2/ReadVariableOp_1:value:0-gru_2/gru_cell_2/strided_slice/stack:output:0/gru_2/gru_cell_2/strided_slice/stack_1:output:0/gru_2/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
gru_2/gru_cell_2/strided_slice?
gru_2/gru_cell_2/MatMulMatMulgru_2/strided_slice_2:output:0'gru_2/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul?
!gru_2/gru_cell_2/ReadVariableOp_2ReadVariableOp*gru_2_gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_2?
&gru_2/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_2/gru_cell_2/strided_slice_1/stack?
(gru_2/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(gru_2/gru_cell_2/strided_slice_1/stack_1?
(gru_2/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_2/gru_cell_2/strided_slice_1/stack_2?
 gru_2/gru_cell_2/strided_slice_1StridedSlice)gru_2/gru_cell_2/ReadVariableOp_2:value:0/gru_2/gru_cell_2/strided_slice_1/stack:output:01gru_2/gru_cell_2/strided_slice_1/stack_1:output:01gru_2/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 gru_2/gru_cell_2/strided_slice_1?
gru_2/gru_cell_2/MatMul_1MatMulgru_2/strided_slice_2:output:0)gru_2/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul_1?
!gru_2/gru_cell_2/ReadVariableOp_3ReadVariableOp*gru_2_gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_3?
&gru_2/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&gru_2/gru_cell_2/strided_slice_2/stack?
(gru_2/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(gru_2/gru_cell_2/strided_slice_2/stack_1?
(gru_2/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_2/gru_cell_2/strided_slice_2/stack_2?
 gru_2/gru_cell_2/strided_slice_2StridedSlice)gru_2/gru_cell_2/ReadVariableOp_3:value:0/gru_2/gru_cell_2/strided_slice_2/stack:output:01gru_2/gru_cell_2/strided_slice_2/stack_1:output:01gru_2/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 gru_2/gru_cell_2/strided_slice_2?
gru_2/gru_cell_2/MatMul_2MatMulgru_2/strided_slice_2:output:0)gru_2/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul_2?
&gru_2/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&gru_2/gru_cell_2/strided_slice_3/stack?
(gru_2/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_2/gru_cell_2/strided_slice_3/stack_1?
(gru_2/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_2/gru_cell_2/strided_slice_3/stack_2?
 gru_2/gru_cell_2/strided_slice_3StridedSlice!gru_2/gru_cell_2/unstack:output:0/gru_2/gru_cell_2/strided_slice_3/stack:output:01gru_2/gru_cell_2/strided_slice_3/stack_1:output:01gru_2/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 gru_2/gru_cell_2/strided_slice_3?
gru_2/gru_cell_2/BiasAddBiasAdd!gru_2/gru_cell_2/MatMul:product:0)gru_2/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd?
&gru_2/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_2/gru_cell_2/strided_slice_4/stack?
(gru_2/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_2/gru_cell_2/strided_slice_4/stack_1?
(gru_2/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_2/gru_cell_2/strided_slice_4/stack_2?
 gru_2/gru_cell_2/strided_slice_4StridedSlice!gru_2/gru_cell_2/unstack:output:0/gru_2/gru_cell_2/strided_slice_4/stack:output:01gru_2/gru_cell_2/strided_slice_4/stack_1:output:01gru_2/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 gru_2/gru_cell_2/strided_slice_4?
gru_2/gru_cell_2/BiasAdd_1BiasAdd#gru_2/gru_cell_2/MatMul_1:product:0)gru_2/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd_1?
&gru_2/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_2/gru_cell_2/strided_slice_5/stack?
(gru_2/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(gru_2/gru_cell_2/strided_slice_5/stack_1?
(gru_2/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_2/gru_cell_2/strided_slice_5/stack_2?
 gru_2/gru_cell_2/strided_slice_5StridedSlice!gru_2/gru_cell_2/unstack:output:0/gru_2/gru_cell_2/strided_slice_5/stack:output:01gru_2/gru_cell_2/strided_slice_5/stack_1:output:01gru_2/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 gru_2/gru_cell_2/strided_slice_5?
gru_2/gru_cell_2/BiasAdd_2BiasAdd#gru_2/gru_cell_2/MatMul_2:product:0)gru_2/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd_2?
gru_2/gru_cell_2/mulMulgru_2/zeros:output:0#gru_2/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul?
gru_2/gru_cell_2/mul_1Mulgru_2/zeros:output:0#gru_2/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul_1?
gru_2/gru_cell_2/mul_2Mulgru_2/zeros:output:0#gru_2/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul_2?
!gru_2/gru_cell_2/ReadVariableOp_4ReadVariableOp*gru_2_gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_4?
&gru_2/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru_2/gru_cell_2/strided_slice_6/stack?
(gru_2/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(gru_2/gru_cell_2/strided_slice_6/stack_1?
(gru_2/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_2/gru_cell_2/strided_slice_6/stack_2?
 gru_2/gru_cell_2/strided_slice_6StridedSlice)gru_2/gru_cell_2/ReadVariableOp_4:value:0/gru_2/gru_cell_2/strided_slice_6/stack:output:01gru_2/gru_cell_2/strided_slice_6/stack_1:output:01gru_2/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 gru_2/gru_cell_2/strided_slice_6?
gru_2/gru_cell_2/MatMul_3MatMulgru_2/gru_cell_2/mul:z:0)gru_2/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul_3?
!gru_2/gru_cell_2/ReadVariableOp_5ReadVariableOp*gru_2_gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_5?
&gru_2/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_2/gru_cell_2/strided_slice_7/stack?
(gru_2/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(gru_2/gru_cell_2/strided_slice_7/stack_1?
(gru_2/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_2/gru_cell_2/strided_slice_7/stack_2?
 gru_2/gru_cell_2/strided_slice_7StridedSlice)gru_2/gru_cell_2/ReadVariableOp_5:value:0/gru_2/gru_cell_2/strided_slice_7/stack:output:01gru_2/gru_cell_2/strided_slice_7/stack_1:output:01gru_2/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 gru_2/gru_cell_2/strided_slice_7?
gru_2/gru_cell_2/MatMul_4MatMulgru_2/gru_cell_2/mul_1:z:0)gru_2/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul_4?
&gru_2/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&gru_2/gru_cell_2/strided_slice_8/stack?
(gru_2/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_2/gru_cell_2/strided_slice_8/stack_1?
(gru_2/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_2/gru_cell_2/strided_slice_8/stack_2?
 gru_2/gru_cell_2/strided_slice_8StridedSlice!gru_2/gru_cell_2/unstack:output:1/gru_2/gru_cell_2/strided_slice_8/stack:output:01gru_2/gru_cell_2/strided_slice_8/stack_1:output:01gru_2/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 gru_2/gru_cell_2/strided_slice_8?
gru_2/gru_cell_2/BiasAdd_3BiasAdd#gru_2/gru_cell_2/MatMul_3:product:0)gru_2/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd_3?
&gru_2/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_2/gru_cell_2/strided_slice_9/stack?
(gru_2/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_2/gru_cell_2/strided_slice_9/stack_1?
(gru_2/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_2/gru_cell_2/strided_slice_9/stack_2?
 gru_2/gru_cell_2/strided_slice_9StridedSlice!gru_2/gru_cell_2/unstack:output:1/gru_2/gru_cell_2/strided_slice_9/stack:output:01gru_2/gru_cell_2/strided_slice_9/stack_1:output:01gru_2/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 gru_2/gru_cell_2/strided_slice_9?
gru_2/gru_cell_2/BiasAdd_4BiasAdd#gru_2/gru_cell_2/MatMul_4:product:0)gru_2/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd_4?
gru_2/gru_cell_2/addAddV2!gru_2/gru_cell_2/BiasAdd:output:0#gru_2/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/add?
gru_2/gru_cell_2/SigmoidSigmoidgru_2/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/Sigmoid?
gru_2/gru_cell_2/add_1AddV2#gru_2/gru_cell_2/BiasAdd_1:output:0#gru_2/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/add_1?
gru_2/gru_cell_2/Sigmoid_1Sigmoidgru_2/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/Sigmoid_1?
!gru_2/gru_cell_2/ReadVariableOp_6ReadVariableOp*gru_2_gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_6?
'gru_2/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'gru_2/gru_cell_2/strided_slice_10/stack?
)gru_2/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)gru_2/gru_cell_2/strided_slice_10/stack_1?
)gru_2/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)gru_2/gru_cell_2/strided_slice_10/stack_2?
!gru_2/gru_cell_2/strided_slice_10StridedSlice)gru_2/gru_cell_2/ReadVariableOp_6:value:00gru_2/gru_cell_2/strided_slice_10/stack:output:02gru_2/gru_cell_2/strided_slice_10/stack_1:output:02gru_2/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!gru_2/gru_cell_2/strided_slice_10?
gru_2/gru_cell_2/MatMul_5MatMulgru_2/gru_cell_2/mul_2:z:0*gru_2/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul_5?
'gru_2/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'gru_2/gru_cell_2/strided_slice_11/stack?
)gru_2/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)gru_2/gru_cell_2/strided_slice_11/stack_1?
)gru_2/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)gru_2/gru_cell_2/strided_slice_11/stack_2?
!gru_2/gru_cell_2/strided_slice_11StridedSlice!gru_2/gru_cell_2/unstack:output:10gru_2/gru_cell_2/strided_slice_11/stack:output:02gru_2/gru_cell_2/strided_slice_11/stack_1:output:02gru_2/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!gru_2/gru_cell_2/strided_slice_11?
gru_2/gru_cell_2/BiasAdd_5BiasAdd#gru_2/gru_cell_2/MatMul_5:product:0*gru_2/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd_5?
gru_2/gru_cell_2/mul_3Mulgru_2/gru_cell_2/Sigmoid_1:y:0#gru_2/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul_3?
gru_2/gru_cell_2/add_2AddV2#gru_2/gru_cell_2/BiasAdd_2:output:0gru_2/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/add_2?
gru_2/gru_cell_2/TanhTanhgru_2/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/Tanh?
gru_2/gru_cell_2/mul_4Mulgru_2/gru_cell_2/Sigmoid:y:0gru_2/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul_4u
gru_2/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_2/gru_cell_2/sub/x?
gru_2/gru_cell_2/subSubgru_2/gru_cell_2/sub/x:output:0gru_2/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/sub?
gru_2/gru_cell_2/mul_5Mulgru_2/gru_cell_2/sub:z:0gru_2/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul_5?
gru_2/gru_cell_2/add_3AddV2gru_2/gru_cell_2/mul_4:z:0gru_2/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/add_3?
#gru_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2%
#gru_2/TensorArrayV2_1/element_shape?
gru_2/TensorArrayV2_1TensorListReserve,gru_2/TensorArrayV2_1/element_shape:output:0gru_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_2/TensorArrayV2_1Z

gru_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_2/time?
gru_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru_2/while/maximum_iterationsv
gru_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_2/while/loop_counter?
gru_2/whileWhile!gru_2/while/loop_counter:output:0'gru_2/while/maximum_iterations:output:0gru_2/time:output:0gru_2/TensorArrayV2_1:handle:0gru_2/zeros:output:0gru_2/strided_slice_1:output:0=gru_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_2_gru_cell_2_readvariableop_resource*gru_2_gru_cell_2_readvariableop_1_resource*gru_2_gru_cell_2_readvariableop_4_resource*
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
gru_2_while_body_565557*#
condR
gru_2_while_cond_565556*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_2/while?
6gru_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   28
6gru_2/TensorArrayV2Stack/TensorListStack/element_shape?
(gru_2/TensorArrayV2Stack/TensorListStackTensorListStackgru_2/while:output:3?gru_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02*
(gru_2/TensorArrayV2Stack/TensorListStack?
gru_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_2/strided_slice_3/stack?
gru_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_2/strided_slice_3/stack_1?
gru_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice_3/stack_2?
gru_2/strided_slice_3StridedSlice1gru_2/TensorArrayV2Stack/TensorListStack:tensor:0$gru_2/strided_slice_3/stack:output:0&gru_2/strided_slice_3/stack_1:output:0&gru_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_2/strided_slice_3?
gru_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_2/transpose_1/perm?
gru_2/transpose_1	Transpose1gru_2/TensorArrayV2Stack/TensorListStack:tensor:0gru_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_2/transpose_1r
gru_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_2/runtime?
layer_normalization_2/ShapeShapegru_2/strided_slice_3:output:0*
T0*
_output_shapes
:2
layer_normalization_2/Shape?
)layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_2/strided_slice/stack?
+layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_2/strided_slice/stack_1?
+layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_2/strided_slice/stack_2?
#layer_normalization_2/strided_sliceStridedSlice$layer_normalization_2/Shape:output:02layer_normalization_2/strided_slice/stack:output:04layer_normalization_2/strided_slice/stack_1:output:04layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_2/strided_slice|
layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_2/mul/x?
layer_normalization_2/mulMul$layer_normalization_2/mul/x:output:0,layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_2/mul?
+layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_2/strided_slice_1/stack?
-layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_2/strided_slice_1/stack_1?
-layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_2/strided_slice_1/stack_2?
%layer_normalization_2/strided_slice_1StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_1/stack:output:06layer_normalization_2/strided_slice_1/stack_1:output:06layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_2/strided_slice_1?
layer_normalization_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_2/mul_1/x?
layer_normalization_2/mul_1Mul&layer_normalization_2/mul_1/x:output:0.layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_2/mul_1?
%layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_2/Reshape/shape/0?
%layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_2/Reshape/shape/3?
#layer_normalization_2/Reshape/shapePack.layer_normalization_2/Reshape/shape/0:output:0layer_normalization_2/mul:z:0layer_normalization_2/mul_1:z:0.layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_2/Reshape/shape?
layer_normalization_2/ReshapeReshapegru_2/strided_slice_3:output:0,layer_normalization_2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_2/Reshape
layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_2/Const?
layer_normalization_2/Fill/dimsPacklayer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_2/Fill/dims?
layer_normalization_2/FillFill(layer_normalization_2/Fill/dims:output:0$layer_normalization_2/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_2/Fill?
layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_2/Const_1?
!layer_normalization_2/Fill_1/dimsPacklayer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_2/Fill_1/dims?
layer_normalization_2/Fill_1Fill*layer_normalization_2/Fill_1/dims:output:0&layer_normalization_2/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_2/Fill_1?
layer_normalization_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_2/Const_2?
layer_normalization_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_2/Const_3?
&layer_normalization_2/FusedBatchNormV3FusedBatchNormV3&layer_normalization_2/Reshape:output:0#layer_normalization_2/Fill:output:0%layer_normalization_2/Fill_1:output:0&layer_normalization_2/Const_2:output:0&layer_normalization_2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_2/FusedBatchNormV3?
layer_normalization_2/Reshape_1Reshape*layer_normalization_2/FusedBatchNormV3:y:0$layer_normalization_2/Shape:output:0*
T0*(
_output_shapes
:??????????2!
layer_normalization_2/Reshape_1?
*layer_normalization_2/mul_2/ReadVariableOpReadVariableOp3layer_normalization_2_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*layer_normalization_2/mul_2/ReadVariableOp?
layer_normalization_2/mul_2Mul(layer_normalization_2/Reshape_1:output:02layer_normalization_2/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_2/mul_2?
(layer_normalization_2/add/ReadVariableOpReadVariableOp1layer_normalization_2_add_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization_2/add/ReadVariableOp?
layer_normalization_2/addAddV2layer_normalization_2/mul_2:z:00layer_normalization_2/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_2/add?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMullayer_normalization_2/add:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd|
dense_12/SigmoidSigmoiddense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_12/Sigmoid?
IdentityIdentitydense_12/Sigmoid:y:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^gru_2/gru_cell_2/ReadVariableOp"^gru_2/gru_cell_2/ReadVariableOp_1"^gru_2/gru_cell_2/ReadVariableOp_2"^gru_2/gru_cell_2/ReadVariableOp_3"^gru_2/gru_cell_2/ReadVariableOp_4"^gru_2/gru_cell_2/ReadVariableOp_5"^gru_2/gru_cell_2/ReadVariableOp_6^gru_2/while)^layer_normalization_2/add/ReadVariableOp+^layer_normalization_2/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
gru_2/gru_cell_2/ReadVariableOpgru_2/gru_cell_2/ReadVariableOp2F
!gru_2/gru_cell_2/ReadVariableOp_1!gru_2/gru_cell_2/ReadVariableOp_12F
!gru_2/gru_cell_2/ReadVariableOp_2!gru_2/gru_cell_2/ReadVariableOp_22F
!gru_2/gru_cell_2/ReadVariableOp_3!gru_2/gru_cell_2/ReadVariableOp_32F
!gru_2/gru_cell_2/ReadVariableOp_4!gru_2/gru_cell_2/ReadVariableOp_42F
!gru_2/gru_cell_2/ReadVariableOp_5!gru_2/gru_cell_2/ReadVariableOp_52F
!gru_2/gru_cell_2/ReadVariableOp_6!gru_2/gru_cell_2/ReadVariableOp_62
gru_2/whilegru_2/while2T
(layer_normalization_2/add/ReadVariableOp(layer_normalization_2/add/ReadVariableOp2X
*layer_normalization_2/mul_2/ReadVariableOp*layer_normalization_2/mul_2/ReadVariableOp:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?	
?
+__inference_gru_cell_2_layer_call_fn_567311

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
*2
config_proto" 

CPU

GPU2*3,4J 8? *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_5636822
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
?	
?
gru_2_while_cond_565556(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1@
<gru_2_while_gru_2_while_cond_565556___redundant_placeholder0@
<gru_2_while_gru_2_while_cond_565556___redundant_placeholder1@
<gru_2_while_gru_2_while_cond_565556___redundant_placeholder2@
<gru_2_while_gru_2_while_cond_565556___redundant_placeholder3
gru_2_while_identity
?
gru_2/while/LessLessgru_2_while_placeholder&gru_2_while_less_gru_2_strided_slice_1*
T0*
_output_shapes
: 2
gru_2/while/Lesso
gru_2/while/IdentityIdentitygru_2/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_2/while/Identity"5
gru_2_while_identitygru_2/while/Identity:output:0*A
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
?!
?
while_body_564037
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_2_564059_0
while_gru_cell_2_564061_0
while_gru_cell_2_564063_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_2_564059
while_gru_cell_2_564061
while_gru_cell_2_564063??(while/gru_cell_2/StatefulPartitionedCall?
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
(while/gru_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_2_564059_0while_gru_cell_2_564061_0while_gru_cell_2_564063_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_5636822*
(while/gru_cell_2/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_2/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity1while/gru_cell_2/StatefulPartitionedCall:output:1)^while/gru_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4"4
while_gru_cell_2_564059while_gru_cell_2_564059_0"4
while_gru_cell_2_564061while_gru_cell_2_564061_0"4
while_gru_cell_2_564063while_gru_cell_2_564063_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2T
(while/gru_cell_2/StatefulPartitionedCall(while/gru_cell_2/StatefulPartitionedCall: 
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
޴
?
A__inference_gru_2_layer_call_and_return_conditional_losses_566988

inputs&
"gru_cell_2_readvariableop_resource(
$gru_cell_2_readvariableop_1_resource(
$gru_cell_2_readvariableop_4_resource
identity??gru_cell_2/ReadVariableOp?gru_cell_2/ReadVariableOp_1?gru_cell_2/ReadVariableOp_2?gru_cell_2/ReadVariableOp_3?gru_cell_2/ReadVariableOp_4?gru_cell_2/ReadVariableOp_5?gru_cell_2/ReadVariableOp_6?whileD
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
gru_cell_2/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/ones_like/Const?
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/ones_like?
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_2/ReadVariableOp?
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_2/unstack?
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_1?
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack?
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice/stack_1?
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2?
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice?
gru_cell_2/MatMulMatMulstrided_slice_2:output:0!gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul?
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_2?
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_1/stack?
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_1/stack_1?
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2?
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_1?
gru_cell_2/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_1?
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_3?
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_2/strided_slice_2/stack?
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1?
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2?
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_2?
gru_cell_2/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_2?
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack?
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_3/stack_1?
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2?
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_3?
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd?
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_4/stack?
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_4/stack_1?
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2?
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_4?
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_1?
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_5/stack?
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1?
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2?
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_5?
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_2?
gru_cell_2/mulMulzeros:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul?
gru_cell_2/mul_1Mulzeros:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_1?
gru_cell_2/mul_2Mulzeros:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_2?
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_4?
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack?
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_2/strided_slice_6/stack_1?
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2?
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_6?
gru_cell_2/MatMul_3MatMulgru_cell_2/mul:z:0#gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_3?
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_5?
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_7/stack?
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_7/stack_1?
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2?
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_7?
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_4?
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack?
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_8/stack_1?
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2?
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_8?
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_3?
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_9/stack?
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_9/stack_1?
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2?
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_9?
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_4?
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/addz
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid?
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_1?
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid_1?
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_6?
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_2/strided_slice_10/stack?
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1?
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2?
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_10?
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_2:z:0$gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_5?
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_2/strided_slice_11/stack?
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1?
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2?
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_11?
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_5?
gru_cell_2/mul_3Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_3?
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_2s
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Tanh?
gru_cell_2/mul_4Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_4i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/sub/x?
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/sub?
gru_cell_2/mul_5Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_5?
gru_cell_2/add_3AddV2gru_cell_2/mul_4:z:0gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
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
while_body_566842*
condR
while_cond_566841*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
A__inference_gru_2_layer_call_and_return_conditional_losses_564550

inputs&
"gru_cell_2_readvariableop_resource(
$gru_cell_2_readvariableop_1_resource(
$gru_cell_2_readvariableop_4_resource
identity??gru_cell_2/ReadVariableOp?gru_cell_2/ReadVariableOp_1?gru_cell_2/ReadVariableOp_2?gru_cell_2/ReadVariableOp_3?gru_cell_2/ReadVariableOp_4?gru_cell_2/ReadVariableOp_5?gru_cell_2/ReadVariableOp_6?whileD
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
gru_cell_2/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/ones_like/Const?
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/ones_likey
gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout/Const?
gru_cell_2/dropout/MulMulgru_cell_2/ones_like:output:0!gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout/Mul?
gru_cell_2/dropout/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout/Shape?
/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???21
/gru_cell_2/dropout/random_uniform/RandomUniform?
!gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell_2/dropout/GreaterEqual/y?
gru_cell_2/dropout/GreaterEqualGreaterEqual8gru_cell_2/dropout/random_uniform/RandomUniform:output:0*gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell_2/dropout/GreaterEqual?
gru_cell_2/dropout/CastCast#gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_2/dropout/Cast?
gru_cell_2/dropout/Mul_1Mulgru_cell_2/dropout/Mul:z:0gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout/Mul_1}
gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_1/Const?
gru_cell_2/dropout_1/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_1/Mul?
gru_cell_2/dropout_1/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_1/Shape?
1gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???23
1gru_cell_2/dropout_1/random_uniform/RandomUniform?
#gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_2/dropout_1/GreaterEqual/y?
!gru_cell_2/dropout_1/GreaterEqualGreaterEqual:gru_cell_2/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_2/dropout_1/GreaterEqual?
gru_cell_2/dropout_1/CastCast%gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_2/dropout_1/Cast?
gru_cell_2/dropout_1/Mul_1Mulgru_cell_2/dropout_1/Mul:z:0gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_1/Mul_1}
gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_2/Const?
gru_cell_2/dropout_2/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_2/Mul?
gru_cell_2/dropout_2/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_2/Shape?
1gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???23
1gru_cell_2/dropout_2/random_uniform/RandomUniform?
#gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_2/dropout_2/GreaterEqual/y?
!gru_cell_2/dropout_2/GreaterEqualGreaterEqual:gru_cell_2/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_2/dropout_2/GreaterEqual?
gru_cell_2/dropout_2/CastCast%gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_2/dropout_2/Cast?
gru_cell_2/dropout_2/Mul_1Mulgru_cell_2/dropout_2/Mul:z:0gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_2/Mul_1?
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_2/ReadVariableOp?
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_2/unstack?
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_1?
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack?
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice/stack_1?
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2?
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice?
gru_cell_2/MatMulMatMulstrided_slice_2:output:0!gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul?
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_2?
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_1/stack?
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_1/stack_1?
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2?
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_1?
gru_cell_2/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_1?
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_3?
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_2/strided_slice_2/stack?
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1?
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2?
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_2?
gru_cell_2/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_2?
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack?
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_3/stack_1?
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2?
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_3?
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd?
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_4/stack?
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_4/stack_1?
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2?
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_4?
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_1?
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_5/stack?
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1?
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2?
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_5?
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_2?
gru_cell_2/mulMulzeros:output:0gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul?
gru_cell_2/mul_1Mulzeros:output:0gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_1?
gru_cell_2/mul_2Mulzeros:output:0gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_2?
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_4?
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack?
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_2/strided_slice_6/stack_1?
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2?
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_6?
gru_cell_2/MatMul_3MatMulgru_cell_2/mul:z:0#gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_3?
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_5?
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_7/stack?
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_7/stack_1?
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2?
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_7?
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_4?
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack?
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_8/stack_1?
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2?
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_8?
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_3?
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_9/stack?
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_9/stack_1?
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2?
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_9?
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_4?
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/addz
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid?
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_1?
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid_1?
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_6?
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_2/strided_slice_10/stack?
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1?
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2?
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_10?
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_2:z:0$gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_5?
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_2/strided_slice_11/stack?
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1?
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2?
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_11?
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_5?
gru_cell_2/mul_3Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_3?
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_2s
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Tanh?
gru_cell_2/mul_4Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_4i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/sub/x?
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/sub?
gru_cell_2/mul_5Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_5?
gru_cell_2/add_3AddV2gru_cell_2/mul_4:z:0gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
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
while_body_564380*
condR
while_cond_564379*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?h
?
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_567297

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
??
?	
gru_2_while_body_565217(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_04
0gru_2_while_gru_cell_2_readvariableop_resource_06
2gru_2_while_gru_cell_2_readvariableop_1_resource_06
2gru_2_while_gru_cell_2_readvariableop_4_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor2
.gru_2_while_gru_cell_2_readvariableop_resource4
0gru_2_while_gru_cell_2_readvariableop_1_resource4
0gru_2_while_gru_cell_2_readvariableop_4_resource??%gru_2/while/gru_cell_2/ReadVariableOp?'gru_2/while/gru_cell_2/ReadVariableOp_1?'gru_2/while/gru_cell_2/ReadVariableOp_2?'gru_2/while/gru_cell_2/ReadVariableOp_3?'gru_2/while/gru_cell_2/ReadVariableOp_4?'gru_2/while/gru_cell_2/ReadVariableOp_5?'gru_2/while/gru_cell_2/ReadVariableOp_6?
=gru_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2?
=gru_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
/gru_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0gru_2_while_placeholderFgru_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype021
/gru_2/while/TensorArrayV2Read/TensorListGetItem?
&gru_2/while/gru_cell_2/ones_like/ShapeShapegru_2_while_placeholder_2*
T0*
_output_shapes
:2(
&gru_2/while/gru_cell_2/ones_like/Shape?
&gru_2/while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&gru_2/while/gru_cell_2/ones_like/Const?
 gru_2/while/gru_cell_2/ones_likeFill/gru_2/while/gru_cell_2/ones_like/Shape:output:0/gru_2/while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/ones_like?
$gru_2/while/gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$gru_2/while/gru_cell_2/dropout/Const?
"gru_2/while/gru_cell_2/dropout/MulMul)gru_2/while/gru_cell_2/ones_like:output:0-gru_2/while/gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2$
"gru_2/while/gru_cell_2/dropout/Mul?
$gru_2/while/gru_cell_2/dropout/ShapeShape)gru_2/while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2&
$gru_2/while/gru_cell_2/dropout/Shape?
;gru_2/while/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform-gru_2/while/gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??M2=
;gru_2/while/gru_cell_2/dropout/random_uniform/RandomUniform?
-gru_2/while/gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-gru_2/while/gru_cell_2/dropout/GreaterEqual/y?
+gru_2/while/gru_cell_2/dropout/GreaterEqualGreaterEqualDgru_2/while/gru_cell_2/dropout/random_uniform/RandomUniform:output:06gru_2/while/gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2-
+gru_2/while/gru_cell_2/dropout/GreaterEqual?
#gru_2/while/gru_cell_2/dropout/CastCast/gru_2/while/gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2%
#gru_2/while/gru_cell_2/dropout/Cast?
$gru_2/while/gru_cell_2/dropout/Mul_1Mul&gru_2/while/gru_cell_2/dropout/Mul:z:0'gru_2/while/gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2&
$gru_2/while/gru_cell_2/dropout/Mul_1?
&gru_2/while/gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&gru_2/while/gru_cell_2/dropout_1/Const?
$gru_2/while/gru_cell_2/dropout_1/MulMul)gru_2/while/gru_cell_2/ones_like:output:0/gru_2/while/gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$gru_2/while/gru_cell_2/dropout_1/Mul?
&gru_2/while/gru_cell_2/dropout_1/ShapeShape)gru_2/while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2(
&gru_2/while/gru_cell_2/dropout_1/Shape?
=gru_2/while/gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform/gru_2/while/gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=gru_2/while/gru_cell_2/dropout_1/random_uniform/RandomUniform?
/gru_2/while/gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/gru_2/while/gru_cell_2/dropout_1/GreaterEqual/y?
-gru_2/while/gru_cell_2/dropout_1/GreaterEqualGreaterEqualFgru_2/while/gru_cell_2/dropout_1/random_uniform/RandomUniform:output:08gru_2/while/gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-gru_2/while/gru_cell_2/dropout_1/GreaterEqual?
%gru_2/while/gru_cell_2/dropout_1/CastCast1gru_2/while/gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%gru_2/while/gru_cell_2/dropout_1/Cast?
&gru_2/while/gru_cell_2/dropout_1/Mul_1Mul(gru_2/while/gru_cell_2/dropout_1/Mul:z:0)gru_2/while/gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&gru_2/while/gru_cell_2/dropout_1/Mul_1?
&gru_2/while/gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&gru_2/while/gru_cell_2/dropout_2/Const?
$gru_2/while/gru_cell_2/dropout_2/MulMul)gru_2/while/gru_cell_2/ones_like:output:0/gru_2/while/gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2&
$gru_2/while/gru_cell_2/dropout_2/Mul?
&gru_2/while/gru_cell_2/dropout_2/ShapeShape)gru_2/while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2(
&gru_2/while/gru_cell_2/dropout_2/Shape?
=gru_2/while/gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform/gru_2/while/gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2?
=gru_2/while/gru_cell_2/dropout_2/random_uniform/RandomUniform?
/gru_2/while/gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/gru_2/while/gru_cell_2/dropout_2/GreaterEqual/y?
-gru_2/while/gru_cell_2/dropout_2/GreaterEqualGreaterEqualFgru_2/while/gru_cell_2/dropout_2/random_uniform/RandomUniform:output:08gru_2/while/gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2/
-gru_2/while/gru_cell_2/dropout_2/GreaterEqual?
%gru_2/while/gru_cell_2/dropout_2/CastCast1gru_2/while/gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2'
%gru_2/while/gru_cell_2/dropout_2/Cast?
&gru_2/while/gru_cell_2/dropout_2/Mul_1Mul(gru_2/while/gru_cell_2/dropout_2/Mul:z:0)gru_2/while/gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2(
&gru_2/while/gru_cell_2/dropout_2/Mul_1?
%gru_2/while/gru_cell_2/ReadVariableOpReadVariableOp0gru_2_while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	?*
dtype02'
%gru_2/while/gru_cell_2/ReadVariableOp?
gru_2/while/gru_cell_2/unstackUnpack-gru_2/while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2 
gru_2/while/gru_cell_2/unstack?
'gru_2/while/gru_cell_2/ReadVariableOp_1ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_1?
*gru_2/while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*gru_2/while/gru_cell_2/strided_slice/stack?
,gru_2/while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_2/while/gru_cell_2/strided_slice/stack_1?
,gru_2/while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,gru_2/while/gru_cell_2/strided_slice/stack_2?
$gru_2/while/gru_cell_2/strided_sliceStridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_1:value:03gru_2/while/gru_cell_2/strided_slice/stack:output:05gru_2/while/gru_cell_2/strided_slice/stack_1:output:05gru_2/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2&
$gru_2/while/gru_cell_2/strided_slice?
gru_2/while/gru_cell_2/MatMulMatMul6gru_2/while/TensorArrayV2Read/TensorListGetItem:item:0-gru_2/while/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/MatMul?
'gru_2/while/gru_cell_2/ReadVariableOp_2ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_2?
,gru_2/while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_2/while/gru_cell_2/strided_slice_1/stack?
.gru_2/while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  20
.gru_2/while/gru_cell_2/strided_slice_1/stack_1?
.gru_2/while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_2/while/gru_cell_2/strided_slice_1/stack_2?
&gru_2/while/gru_cell_2/strided_slice_1StridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_2:value:05gru_2/while/gru_cell_2/strided_slice_1/stack:output:07gru_2/while/gru_cell_2/strided_slice_1/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&gru_2/while/gru_cell_2/strided_slice_1?
gru_2/while/gru_cell_2/MatMul_1MatMul6gru_2/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_2/while/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2!
gru_2/while/gru_cell_2/MatMul_1?
'gru_2/while/gru_cell_2/ReadVariableOp_3ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_3?
,gru_2/while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2.
,gru_2/while/gru_cell_2/strided_slice_2/stack?
.gru_2/while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.gru_2/while/gru_cell_2/strided_slice_2/stack_1?
.gru_2/while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_2/while/gru_cell_2/strided_slice_2/stack_2?
&gru_2/while/gru_cell_2/strided_slice_2StridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_3:value:05gru_2/while/gru_cell_2/strided_slice_2/stack:output:07gru_2/while/gru_cell_2/strided_slice_2/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&gru_2/while/gru_cell_2/strided_slice_2?
gru_2/while/gru_cell_2/MatMul_2MatMul6gru_2/while/TensorArrayV2Read/TensorListGetItem:item:0/gru_2/while/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2!
gru_2/while/gru_cell_2/MatMul_2?
,gru_2/while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,gru_2/while/gru_cell_2/strided_slice_3/stack?
.gru_2/while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_2/while/gru_cell_2/strided_slice_3/stack_1?
.gru_2/while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_2/while/gru_cell_2/strided_slice_3/stack_2?
&gru_2/while/gru_cell_2/strided_slice_3StridedSlice'gru_2/while/gru_cell_2/unstack:output:05gru_2/while/gru_cell_2/strided_slice_3/stack:output:07gru_2/while/gru_cell_2/strided_slice_3/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2(
&gru_2/while/gru_cell_2/strided_slice_3?
gru_2/while/gru_cell_2/BiasAddBiasAdd'gru_2/while/gru_cell_2/MatMul:product:0/gru_2/while/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2 
gru_2/while/gru_cell_2/BiasAdd?
,gru_2/while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_2/while/gru_cell_2/strided_slice_4/stack?
.gru_2/while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_2/while/gru_cell_2/strided_slice_4/stack_1?
.gru_2/while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_2/while/gru_cell_2/strided_slice_4/stack_2?
&gru_2/while/gru_cell_2/strided_slice_4StridedSlice'gru_2/while/gru_cell_2/unstack:output:05gru_2/while/gru_cell_2/strided_slice_4/stack:output:07gru_2/while/gru_cell_2/strided_slice_4/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2(
&gru_2/while/gru_cell_2/strided_slice_4?
 gru_2/while/gru_cell_2/BiasAdd_1BiasAdd)gru_2/while/gru_cell_2/MatMul_1:product:0/gru_2/while/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/BiasAdd_1?
,gru_2/while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_2/while/gru_cell_2/strided_slice_5/stack?
.gru_2/while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.gru_2/while/gru_cell_2/strided_slice_5/stack_1?
.gru_2/while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_2/while/gru_cell_2/strided_slice_5/stack_2?
&gru_2/while/gru_cell_2/strided_slice_5StridedSlice'gru_2/while/gru_cell_2/unstack:output:05gru_2/while/gru_cell_2/strided_slice_5/stack:output:07gru_2/while/gru_cell_2/strided_slice_5/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2(
&gru_2/while/gru_cell_2/strided_slice_5?
 gru_2/while/gru_cell_2/BiasAdd_2BiasAdd)gru_2/while/gru_cell_2/MatMul_2:product:0/gru_2/while/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/BiasAdd_2?
gru_2/while/gru_cell_2/mulMulgru_2_while_placeholder_2(gru_2/while/gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul?
gru_2/while/gru_cell_2/mul_1Mulgru_2_while_placeholder_2*gru_2/while/gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul_1?
gru_2/while/gru_cell_2/mul_2Mulgru_2_while_placeholder_2*gru_2/while/gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul_2?
'gru_2/while/gru_cell_2/ReadVariableOp_4ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_4?
,gru_2/while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,gru_2/while/gru_cell_2/strided_slice_6/stack?
.gru_2/while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   20
.gru_2/while/gru_cell_2/strided_slice_6/stack_1?
.gru_2/while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_2/while/gru_cell_2/strided_slice_6/stack_2?
&gru_2/while/gru_cell_2/strided_slice_6StridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_4:value:05gru_2/while/gru_cell_2/strided_slice_6/stack:output:07gru_2/while/gru_cell_2/strided_slice_6/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&gru_2/while/gru_cell_2/strided_slice_6?
gru_2/while/gru_cell_2/MatMul_3MatMulgru_2/while/gru_cell_2/mul:z:0/gru_2/while/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2!
gru_2/while/gru_cell_2/MatMul_3?
'gru_2/while/gru_cell_2/ReadVariableOp_5ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_5?
,gru_2/while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2.
,gru_2/while/gru_cell_2/strided_slice_7/stack?
.gru_2/while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  20
.gru_2/while/gru_cell_2/strided_slice_7/stack_1?
.gru_2/while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.gru_2/while/gru_cell_2/strided_slice_7/stack_2?
&gru_2/while/gru_cell_2/strided_slice_7StridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_5:value:05gru_2/while/gru_cell_2/strided_slice_7/stack:output:07gru_2/while/gru_cell_2/strided_slice_7/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&gru_2/while/gru_cell_2/strided_slice_7?
gru_2/while/gru_cell_2/MatMul_4MatMul gru_2/while/gru_cell_2/mul_1:z:0/gru_2/while/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2!
gru_2/while/gru_cell_2/MatMul_4?
,gru_2/while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,gru_2/while/gru_cell_2/strided_slice_8/stack?
.gru_2/while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_2/while/gru_cell_2/strided_slice_8/stack_1?
.gru_2/while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_2/while/gru_cell_2/strided_slice_8/stack_2?
&gru_2/while/gru_cell_2/strided_slice_8StridedSlice'gru_2/while/gru_cell_2/unstack:output:15gru_2/while/gru_cell_2/strided_slice_8/stack:output:07gru_2/while/gru_cell_2/strided_slice_8/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2(
&gru_2/while/gru_cell_2/strided_slice_8?
 gru_2/while/gru_cell_2/BiasAdd_3BiasAdd)gru_2/while/gru_cell_2/MatMul_3:product:0/gru_2/while/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/BiasAdd_3?
,gru_2/while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2.
,gru_2/while/gru_cell_2/strided_slice_9/stack?
.gru_2/while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?20
.gru_2/while/gru_cell_2/strided_slice_9/stack_1?
.gru_2/while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.gru_2/while/gru_cell_2/strided_slice_9/stack_2?
&gru_2/while/gru_cell_2/strided_slice_9StridedSlice'gru_2/while/gru_cell_2/unstack:output:15gru_2/while/gru_cell_2/strided_slice_9/stack:output:07gru_2/while/gru_cell_2/strided_slice_9/stack_1:output:07gru_2/while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2(
&gru_2/while/gru_cell_2/strided_slice_9?
 gru_2/while/gru_cell_2/BiasAdd_4BiasAdd)gru_2/while/gru_cell_2/MatMul_4:product:0/gru_2/while/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/BiasAdd_4?
gru_2/while/gru_cell_2/addAddV2'gru_2/while/gru_cell_2/BiasAdd:output:0)gru_2/while/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/add?
gru_2/while/gru_cell_2/SigmoidSigmoidgru_2/while/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2 
gru_2/while/gru_cell_2/Sigmoid?
gru_2/while/gru_cell_2/add_1AddV2)gru_2/while/gru_cell_2/BiasAdd_1:output:0)gru_2/while/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/add_1?
 gru_2/while/gru_cell_2/Sigmoid_1Sigmoid gru_2/while/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/Sigmoid_1?
'gru_2/while/gru_cell_2/ReadVariableOp_6ReadVariableOp2gru_2_while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02)
'gru_2/while/gru_cell_2/ReadVariableOp_6?
-gru_2/while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2/
-gru_2/while/gru_cell_2/strided_slice_10/stack?
/gru_2/while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/gru_2/while/gru_cell_2/strided_slice_10/stack_1?
/gru_2/while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/gru_2/while/gru_cell_2/strided_slice_10/stack_2?
'gru_2/while/gru_cell_2/strided_slice_10StridedSlice/gru_2/while/gru_cell_2/ReadVariableOp_6:value:06gru_2/while/gru_cell_2/strided_slice_10/stack:output:08gru_2/while/gru_cell_2/strided_slice_10/stack_1:output:08gru_2/while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2)
'gru_2/while/gru_cell_2/strided_slice_10?
gru_2/while/gru_cell_2/MatMul_5MatMul gru_2/while/gru_cell_2/mul_2:z:00gru_2/while/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2!
gru_2/while/gru_cell_2/MatMul_5?
-gru_2/while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2/
-gru_2/while/gru_cell_2/strided_slice_11/stack?
/gru_2/while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/gru_2/while/gru_cell_2/strided_slice_11/stack_1?
/gru_2/while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/gru_2/while/gru_cell_2/strided_slice_11/stack_2?
'gru_2/while/gru_cell_2/strided_slice_11StridedSlice'gru_2/while/gru_cell_2/unstack:output:16gru_2/while/gru_cell_2/strided_slice_11/stack:output:08gru_2/while/gru_cell_2/strided_slice_11/stack_1:output:08gru_2/while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2)
'gru_2/while/gru_cell_2/strided_slice_11?
 gru_2/while/gru_cell_2/BiasAdd_5BiasAdd)gru_2/while/gru_cell_2/MatMul_5:product:00gru_2/while/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2"
 gru_2/while/gru_cell_2/BiasAdd_5?
gru_2/while/gru_cell_2/mul_3Mul$gru_2/while/gru_cell_2/Sigmoid_1:y:0)gru_2/while/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul_3?
gru_2/while/gru_cell_2/add_2AddV2)gru_2/while/gru_cell_2/BiasAdd_2:output:0 gru_2/while/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/add_2?
gru_2/while/gru_cell_2/TanhTanh gru_2/while/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/Tanh?
gru_2/while/gru_cell_2/mul_4Mul"gru_2/while/gru_cell_2/Sigmoid:y:0gru_2_while_placeholder_2*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul_4?
gru_2/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_2/while/gru_cell_2/sub/x?
gru_2/while/gru_cell_2/subSub%gru_2/while/gru_cell_2/sub/x:output:0"gru_2/while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/sub?
gru_2/while/gru_cell_2/mul_5Mulgru_2/while/gru_cell_2/sub:z:0gru_2/while/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/mul_5?
gru_2/while/gru_cell_2/add_3AddV2 gru_2/while/gru_cell_2/mul_4:z:0 gru_2/while/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_2/while/gru_cell_2/add_3?
0gru_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder gru_2/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_2/while/TensorArrayV2Write/TensorListSetItemh
gru_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_2/while/add/y?
gru_2/while/addAddV2gru_2_while_placeholdergru_2/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_2/while/addl
gru_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_2/while/add_1/y?
gru_2/while/add_1AddV2$gru_2_while_gru_2_while_loop_countergru_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_2/while/add_1?
gru_2/while/IdentityIdentitygru_2/while/add_1:z:0&^gru_2/while/gru_cell_2/ReadVariableOp(^gru_2/while/gru_cell_2/ReadVariableOp_1(^gru_2/while/gru_cell_2/ReadVariableOp_2(^gru_2/while/gru_cell_2/ReadVariableOp_3(^gru_2/while/gru_cell_2/ReadVariableOp_4(^gru_2/while/gru_cell_2/ReadVariableOp_5(^gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_2/while/Identity?
gru_2/while/Identity_1Identity*gru_2_while_gru_2_while_maximum_iterations&^gru_2/while/gru_cell_2/ReadVariableOp(^gru_2/while/gru_cell_2/ReadVariableOp_1(^gru_2/while/gru_cell_2/ReadVariableOp_2(^gru_2/while/gru_cell_2/ReadVariableOp_3(^gru_2/while/gru_cell_2/ReadVariableOp_4(^gru_2/while/gru_cell_2/ReadVariableOp_5(^gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_2/while/Identity_1?
gru_2/while/Identity_2Identitygru_2/while/add:z:0&^gru_2/while/gru_cell_2/ReadVariableOp(^gru_2/while/gru_cell_2/ReadVariableOp_1(^gru_2/while/gru_cell_2/ReadVariableOp_2(^gru_2/while/gru_cell_2/ReadVariableOp_3(^gru_2/while/gru_cell_2/ReadVariableOp_4(^gru_2/while/gru_cell_2/ReadVariableOp_5(^gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_2/while/Identity_2?
gru_2/while/Identity_3Identity@gru_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^gru_2/while/gru_cell_2/ReadVariableOp(^gru_2/while/gru_cell_2/ReadVariableOp_1(^gru_2/while/gru_cell_2/ReadVariableOp_2(^gru_2/while/gru_cell_2/ReadVariableOp_3(^gru_2/while/gru_cell_2/ReadVariableOp_4(^gru_2/while/gru_cell_2/ReadVariableOp_5(^gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru_2/while/Identity_3?
gru_2/while/Identity_4Identity gru_2/while/gru_cell_2/add_3:z:0&^gru_2/while/gru_cell_2/ReadVariableOp(^gru_2/while/gru_cell_2/ReadVariableOp_1(^gru_2/while/gru_cell_2/ReadVariableOp_2(^gru_2/while/gru_cell_2/ReadVariableOp_3(^gru_2/while/gru_cell_2/ReadVariableOp_4(^gru_2/while/gru_cell_2/ReadVariableOp_5(^gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
gru_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"f
0gru_2_while_gru_cell_2_readvariableop_1_resource2gru_2_while_gru_cell_2_readvariableop_1_resource_0"f
0gru_2_while_gru_cell_2_readvariableop_4_resource2gru_2_while_gru_cell_2_readvariableop_4_resource_0"b
.gru_2_while_gru_cell_2_readvariableop_resource0gru_2_while_gru_cell_2_readvariableop_resource_0"5
gru_2_while_identitygru_2/while/Identity:output:0"9
gru_2_while_identity_1gru_2/while/Identity_1:output:0"9
gru_2_while_identity_2gru_2/while/Identity_2:output:0"9
gru_2_while_identity_3gru_2/while/Identity_3:output:0"9
gru_2_while_identity_4gru_2/while/Identity_4:output:0"?
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2N
%gru_2/while/gru_cell_2/ReadVariableOp%gru_2/while/gru_cell_2/ReadVariableOp2R
'gru_2/while/gru_cell_2/ReadVariableOp_1'gru_2/while/gru_cell_2/ReadVariableOp_12R
'gru_2/while/gru_cell_2/ReadVariableOp_2'gru_2/while/gru_cell_2/ReadVariableOp_22R
'gru_2/while/gru_cell_2/ReadVariableOp_3'gru_2/while/gru_cell_2/ReadVariableOp_32R
'gru_2/while/gru_cell_2/ReadVariableOp_4'gru_2/while/gru_cell_2/ReadVariableOp_42R
'gru_2/while/gru_cell_2/ReadVariableOp_5'gru_2/while/gru_cell_2/ReadVariableOp_52R
'gru_2/while/gru_cell_2/ReadVariableOp_6'gru_2/while/gru_cell_2/ReadVariableOp_6: 
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
D__inference_dense_12_layer_call_and_return_conditional_losses_567072

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
?
&__inference_gru_2_layer_call_fn_566999

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
*2
config_proto" 

CPU

GPU2*3,4J 8? *J
fERC
A__inference_gru_2_layer_call_and_return_conditional_losses_5645502
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
?
?
&__inference_gru_2_layer_call_fn_566387
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
*2
config_proto" 

CPU

GPU2*3,4J 8? *J
fERC
A__inference_gru_2_layer_call_and_return_conditional_losses_5641012
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
?
?
(__inference_model_4_layer_call_fn_565786

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*2
config_proto" 

CPU

GPU2*3,4J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_5650222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
(__inference_model_4_layer_call_fn_565767

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*2
config_proto" 

CPU

GPU2*3,4J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_5649822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?	
?
gru_2_while_cond_565216(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1@
<gru_2_while_gru_2_while_cond_565216___redundant_placeholder0@
<gru_2_while_gru_2_while_cond_565216___redundant_placeholder1@
<gru_2_while_gru_2_while_cond_565216___redundant_placeholder2@
<gru_2_while_gru_2_while_cond_565216___redundant_placeholder3
gru_2_while_identity
?
gru_2/while/LessLessgru_2_while_placeholder&gru_2_while_less_gru_2_strided_slice_1*
T0*
_output_shapes
: 2
gru_2/while/Lesso
gru_2/while/IdentityIdentitygru_2/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_2/while/Identity"5
gru_2_while_identitygru_2/while/Identity:output:0*A
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
~
)__inference_dense_12_layer_call_fn_567081

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
*2
config_proto" 

CPU

GPU2*3,4J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5649202
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
?
?
C__inference_model_4_layer_call_and_return_conditional_losses_565022

inputs
gru_2_565004
gru_2_565006
gru_2_565008 
layer_normalization_2_565011 
layer_normalization_2_565013
dense_12_565016
dense_12_565018
identity?? dense_12/StatefulPartitionedCall?gru_2/StatefulPartitionedCall?-layer_normalization_2/StatefulPartitionedCall?
gru_2/StatefulPartitionedCallStatefulPartitionedCallinputsgru_2_565004gru_2_565006gru_2_565008*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *J
fERC
A__inference_gru_2_layer_call_and_return_conditional_losses_5648212
gru_2/StatefulPartitionedCall?
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&gru_2/StatefulPartitionedCall:output:0layer_normalization_2_565011layer_normalization_2_565013*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_5648932/
-layer_normalization_2/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_12_565016dense_12_565018*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5649202"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall^gru_2/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2>
gru_2/StatefulPartitionedCallgru_2/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
C__inference_model_4_layer_call_and_return_conditional_losses_565432

inputs,
(gru_2_gru_cell_2_readvariableop_resource.
*gru_2_gru_cell_2_readvariableop_1_resource.
*gru_2_gru_cell_2_readvariableop_4_resource7
3layer_normalization_2_mul_2_readvariableop_resource5
1layer_normalization_2_add_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?gru_2/gru_cell_2/ReadVariableOp?!gru_2/gru_cell_2/ReadVariableOp_1?!gru_2/gru_cell_2/ReadVariableOp_2?!gru_2/gru_cell_2/ReadVariableOp_3?!gru_2/gru_cell_2/ReadVariableOp_4?!gru_2/gru_cell_2/ReadVariableOp_5?!gru_2/gru_cell_2/ReadVariableOp_6?gru_2/while?(layer_normalization_2/add/ReadVariableOp?*layer_normalization_2/mul_2/ReadVariableOpP
gru_2/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_2/Shape?
gru_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_2/strided_slice/stack?
gru_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice/stack_1?
gru_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice/stack_2?
gru_2/strided_sliceStridedSlicegru_2/Shape:output:0"gru_2/strided_slice/stack:output:0$gru_2/strided_slice/stack_1:output:0$gru_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_2/strided_slicei
gru_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_2/zeros/mul/y?
gru_2/zeros/mulMulgru_2/strided_slice:output:0gru_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_2/zeros/mulk
gru_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_2/zeros/Less/y
gru_2/zeros/LessLessgru_2/zeros/mul:z:0gru_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_2/zeros/Lesso
gru_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_2/zeros/packed/1?
gru_2/zeros/packedPackgru_2/strided_slice:output:0gru_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_2/zeros/packedk
gru_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_2/zeros/Const?
gru_2/zerosFillgru_2/zeros/packed:output:0gru_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_2/zeros?
gru_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_2/transpose/perm?
gru_2/transpose	Transposeinputsgru_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????	2
gru_2/transposea
gru_2/Shape_1Shapegru_2/transpose:y:0*
T0*
_output_shapes
:2
gru_2/Shape_1?
gru_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_2/strided_slice_1/stack?
gru_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice_1/stack_1?
gru_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice_1/stack_2?
gru_2/strided_slice_1StridedSlicegru_2/Shape_1:output:0$gru_2/strided_slice_1/stack:output:0&gru_2/strided_slice_1/stack_1:output:0&gru_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_2/strided_slice_1?
!gru_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru_2/TensorArrayV2/element_shape?
gru_2/TensorArrayV2TensorListReserve*gru_2/TensorArrayV2/element_shape:output:0gru_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_2/TensorArrayV2?
;gru_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2=
;gru_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
-gru_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_2/transpose:y:0Dgru_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_2/TensorArrayUnstack/TensorListFromTensor?
gru_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_2/strided_slice_2/stack?
gru_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice_2/stack_1?
gru_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice_2/stack_2?
gru_2/strided_slice_2StridedSlicegru_2/transpose:y:0$gru_2/strided_slice_2/stack:output:0&gru_2/strided_slice_2/stack_1:output:0&gru_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
gru_2/strided_slice_2?
 gru_2/gru_cell_2/ones_like/ShapeShapegru_2/zeros:output:0*
T0*
_output_shapes
:2"
 gru_2/gru_cell_2/ones_like/Shape?
 gru_2/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 gru_2/gru_cell_2/ones_like/Const?
gru_2/gru_cell_2/ones_likeFill)gru_2/gru_cell_2/ones_like/Shape:output:0)gru_2/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/ones_like?
gru_2/gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
gru_2/gru_cell_2/dropout/Const?
gru_2/gru_cell_2/dropout/MulMul#gru_2/gru_cell_2/ones_like:output:0'gru_2/gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/dropout/Mul?
gru_2/gru_cell_2/dropout/ShapeShape#gru_2/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2 
gru_2/gru_cell_2/dropout/Shape?
5gru_2/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform'gru_2/gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??F27
5gru_2/gru_cell_2/dropout/random_uniform/RandomUniform?
'gru_2/gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'gru_2/gru_cell_2/dropout/GreaterEqual/y?
%gru_2/gru_cell_2/dropout/GreaterEqualGreaterEqual>gru_2/gru_cell_2/dropout/random_uniform/RandomUniform:output:00gru_2/gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%gru_2/gru_cell_2/dropout/GreaterEqual?
gru_2/gru_cell_2/dropout/CastCast)gru_2/gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_2/gru_cell_2/dropout/Cast?
gru_2/gru_cell_2/dropout/Mul_1Mul gru_2/gru_cell_2/dropout/Mul:z:0!gru_2/gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2 
gru_2/gru_cell_2/dropout/Mul_1?
 gru_2/gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 gru_2/gru_cell_2/dropout_1/Const?
gru_2/gru_cell_2/dropout_1/MulMul#gru_2/gru_cell_2/ones_like:output:0)gru_2/gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
gru_2/gru_cell_2/dropout_1/Mul?
 gru_2/gru_cell_2/dropout_1/ShapeShape#gru_2/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 gru_2/gru_cell_2/dropout_1/Shape?
7gru_2/gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform)gru_2/gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??;29
7gru_2/gru_cell_2/dropout_1/random_uniform/RandomUniform?
)gru_2/gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)gru_2/gru_cell_2/dropout_1/GreaterEqual/y?
'gru_2/gru_cell_2/dropout_1/GreaterEqualGreaterEqual@gru_2/gru_cell_2/dropout_1/random_uniform/RandomUniform:output:02gru_2/gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'gru_2/gru_cell_2/dropout_1/GreaterEqual?
gru_2/gru_cell_2/dropout_1/CastCast+gru_2/gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
gru_2/gru_cell_2/dropout_1/Cast?
 gru_2/gru_cell_2/dropout_1/Mul_1Mul"gru_2/gru_cell_2/dropout_1/Mul:z:0#gru_2/gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 gru_2/gru_cell_2/dropout_1/Mul_1?
 gru_2/gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 gru_2/gru_cell_2/dropout_2/Const?
gru_2/gru_cell_2/dropout_2/MulMul#gru_2/gru_cell_2/ones_like:output:0)gru_2/gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
gru_2/gru_cell_2/dropout_2/Mul?
 gru_2/gru_cell_2/dropout_2/ShapeShape#gru_2/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 gru_2/gru_cell_2/dropout_2/Shape?
7gru_2/gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform)gru_2/gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??)29
7gru_2/gru_cell_2/dropout_2/random_uniform/RandomUniform?
)gru_2/gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)gru_2/gru_cell_2/dropout_2/GreaterEqual/y?
'gru_2/gru_cell_2/dropout_2/GreaterEqualGreaterEqual@gru_2/gru_cell_2/dropout_2/random_uniform/RandomUniform:output:02gru_2/gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'gru_2/gru_cell_2/dropout_2/GreaterEqual?
gru_2/gru_cell_2/dropout_2/CastCast+gru_2/gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
gru_2/gru_cell_2/dropout_2/Cast?
 gru_2/gru_cell_2/dropout_2/Mul_1Mul"gru_2/gru_cell_2/dropout_2/Mul:z:0#gru_2/gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 gru_2/gru_cell_2/dropout_2/Mul_1?
gru_2/gru_cell_2/ReadVariableOpReadVariableOp(gru_2_gru_cell_2_readvariableop_resource*
_output_shapes
:	?*
dtype02!
gru_2/gru_cell_2/ReadVariableOp?
gru_2/gru_cell_2/unstackUnpack'gru_2/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_2/gru_cell_2/unstack?
!gru_2/gru_cell_2/ReadVariableOp_1ReadVariableOp*gru_2_gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_1?
$gru_2/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$gru_2/gru_cell_2/strided_slice/stack?
&gru_2/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_2/gru_cell_2/strided_slice/stack_1?
&gru_2/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&gru_2/gru_cell_2/strided_slice/stack_2?
gru_2/gru_cell_2/strided_sliceStridedSlice)gru_2/gru_cell_2/ReadVariableOp_1:value:0-gru_2/gru_cell_2/strided_slice/stack:output:0/gru_2/gru_cell_2/strided_slice/stack_1:output:0/gru_2/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
gru_2/gru_cell_2/strided_slice?
gru_2/gru_cell_2/MatMulMatMulgru_2/strided_slice_2:output:0'gru_2/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul?
!gru_2/gru_cell_2/ReadVariableOp_2ReadVariableOp*gru_2_gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_2?
&gru_2/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_2/gru_cell_2/strided_slice_1/stack?
(gru_2/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(gru_2/gru_cell_2/strided_slice_1/stack_1?
(gru_2/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_2/gru_cell_2/strided_slice_1/stack_2?
 gru_2/gru_cell_2/strided_slice_1StridedSlice)gru_2/gru_cell_2/ReadVariableOp_2:value:0/gru_2/gru_cell_2/strided_slice_1/stack:output:01gru_2/gru_cell_2/strided_slice_1/stack_1:output:01gru_2/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 gru_2/gru_cell_2/strided_slice_1?
gru_2/gru_cell_2/MatMul_1MatMulgru_2/strided_slice_2:output:0)gru_2/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul_1?
!gru_2/gru_cell_2/ReadVariableOp_3ReadVariableOp*gru_2_gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_3?
&gru_2/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&gru_2/gru_cell_2/strided_slice_2/stack?
(gru_2/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(gru_2/gru_cell_2/strided_slice_2/stack_1?
(gru_2/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_2/gru_cell_2/strided_slice_2/stack_2?
 gru_2/gru_cell_2/strided_slice_2StridedSlice)gru_2/gru_cell_2/ReadVariableOp_3:value:0/gru_2/gru_cell_2/strided_slice_2/stack:output:01gru_2/gru_cell_2/strided_slice_2/stack_1:output:01gru_2/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 gru_2/gru_cell_2/strided_slice_2?
gru_2/gru_cell_2/MatMul_2MatMulgru_2/strided_slice_2:output:0)gru_2/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul_2?
&gru_2/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&gru_2/gru_cell_2/strided_slice_3/stack?
(gru_2/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_2/gru_cell_2/strided_slice_3/stack_1?
(gru_2/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_2/gru_cell_2/strided_slice_3/stack_2?
 gru_2/gru_cell_2/strided_slice_3StridedSlice!gru_2/gru_cell_2/unstack:output:0/gru_2/gru_cell_2/strided_slice_3/stack:output:01gru_2/gru_cell_2/strided_slice_3/stack_1:output:01gru_2/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 gru_2/gru_cell_2/strided_slice_3?
gru_2/gru_cell_2/BiasAddBiasAdd!gru_2/gru_cell_2/MatMul:product:0)gru_2/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd?
&gru_2/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_2/gru_cell_2/strided_slice_4/stack?
(gru_2/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_2/gru_cell_2/strided_slice_4/stack_1?
(gru_2/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_2/gru_cell_2/strided_slice_4/stack_2?
 gru_2/gru_cell_2/strided_slice_4StridedSlice!gru_2/gru_cell_2/unstack:output:0/gru_2/gru_cell_2/strided_slice_4/stack:output:01gru_2/gru_cell_2/strided_slice_4/stack_1:output:01gru_2/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 gru_2/gru_cell_2/strided_slice_4?
gru_2/gru_cell_2/BiasAdd_1BiasAdd#gru_2/gru_cell_2/MatMul_1:product:0)gru_2/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd_1?
&gru_2/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_2/gru_cell_2/strided_slice_5/stack?
(gru_2/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(gru_2/gru_cell_2/strided_slice_5/stack_1?
(gru_2/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_2/gru_cell_2/strided_slice_5/stack_2?
 gru_2/gru_cell_2/strided_slice_5StridedSlice!gru_2/gru_cell_2/unstack:output:0/gru_2/gru_cell_2/strided_slice_5/stack:output:01gru_2/gru_cell_2/strided_slice_5/stack_1:output:01gru_2/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 gru_2/gru_cell_2/strided_slice_5?
gru_2/gru_cell_2/BiasAdd_2BiasAdd#gru_2/gru_cell_2/MatMul_2:product:0)gru_2/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd_2?
gru_2/gru_cell_2/mulMulgru_2/zeros:output:0"gru_2/gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul?
gru_2/gru_cell_2/mul_1Mulgru_2/zeros:output:0$gru_2/gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul_1?
gru_2/gru_cell_2/mul_2Mulgru_2/zeros:output:0$gru_2/gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul_2?
!gru_2/gru_cell_2/ReadVariableOp_4ReadVariableOp*gru_2_gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_4?
&gru_2/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru_2/gru_cell_2/strided_slice_6/stack?
(gru_2/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(gru_2/gru_cell_2/strided_slice_6/stack_1?
(gru_2/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_2/gru_cell_2/strided_slice_6/stack_2?
 gru_2/gru_cell_2/strided_slice_6StridedSlice)gru_2/gru_cell_2/ReadVariableOp_4:value:0/gru_2/gru_cell_2/strided_slice_6/stack:output:01gru_2/gru_cell_2/strided_slice_6/stack_1:output:01gru_2/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 gru_2/gru_cell_2/strided_slice_6?
gru_2/gru_cell_2/MatMul_3MatMulgru_2/gru_cell_2/mul:z:0)gru_2/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul_3?
!gru_2/gru_cell_2/ReadVariableOp_5ReadVariableOp*gru_2_gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_5?
&gru_2/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&gru_2/gru_cell_2/strided_slice_7/stack?
(gru_2/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(gru_2/gru_cell_2/strided_slice_7/stack_1?
(gru_2/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru_2/gru_cell_2/strided_slice_7/stack_2?
 gru_2/gru_cell_2/strided_slice_7StridedSlice)gru_2/gru_cell_2/ReadVariableOp_5:value:0/gru_2/gru_cell_2/strided_slice_7/stack:output:01gru_2/gru_cell_2/strided_slice_7/stack_1:output:01gru_2/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 gru_2/gru_cell_2/strided_slice_7?
gru_2/gru_cell_2/MatMul_4MatMulgru_2/gru_cell_2/mul_1:z:0)gru_2/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul_4?
&gru_2/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&gru_2/gru_cell_2/strided_slice_8/stack?
(gru_2/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_2/gru_cell_2/strided_slice_8/stack_1?
(gru_2/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_2/gru_cell_2/strided_slice_8/stack_2?
 gru_2/gru_cell_2/strided_slice_8StridedSlice!gru_2/gru_cell_2/unstack:output:1/gru_2/gru_cell_2/strided_slice_8/stack:output:01gru_2/gru_cell_2/strided_slice_8/stack_1:output:01gru_2/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 gru_2/gru_cell_2/strided_slice_8?
gru_2/gru_cell_2/BiasAdd_3BiasAdd#gru_2/gru_cell_2/MatMul_3:product:0)gru_2/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd_3?
&gru_2/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&gru_2/gru_cell_2/strided_slice_9/stack?
(gru_2/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(gru_2/gru_cell_2/strided_slice_9/stack_1?
(gru_2/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(gru_2/gru_cell_2/strided_slice_9/stack_2?
 gru_2/gru_cell_2/strided_slice_9StridedSlice!gru_2/gru_cell_2/unstack:output:1/gru_2/gru_cell_2/strided_slice_9/stack:output:01gru_2/gru_cell_2/strided_slice_9/stack_1:output:01gru_2/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 gru_2/gru_cell_2/strided_slice_9?
gru_2/gru_cell_2/BiasAdd_4BiasAdd#gru_2/gru_cell_2/MatMul_4:product:0)gru_2/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd_4?
gru_2/gru_cell_2/addAddV2!gru_2/gru_cell_2/BiasAdd:output:0#gru_2/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/add?
gru_2/gru_cell_2/SigmoidSigmoidgru_2/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/Sigmoid?
gru_2/gru_cell_2/add_1AddV2#gru_2/gru_cell_2/BiasAdd_1:output:0#gru_2/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/add_1?
gru_2/gru_cell_2/Sigmoid_1Sigmoidgru_2/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/Sigmoid_1?
!gru_2/gru_cell_2/ReadVariableOp_6ReadVariableOp*gru_2_gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02#
!gru_2/gru_cell_2/ReadVariableOp_6?
'gru_2/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'gru_2/gru_cell_2/strided_slice_10/stack?
)gru_2/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)gru_2/gru_cell_2/strided_slice_10/stack_1?
)gru_2/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)gru_2/gru_cell_2/strided_slice_10/stack_2?
!gru_2/gru_cell_2/strided_slice_10StridedSlice)gru_2/gru_cell_2/ReadVariableOp_6:value:00gru_2/gru_cell_2/strided_slice_10/stack:output:02gru_2/gru_cell_2/strided_slice_10/stack_1:output:02gru_2/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!gru_2/gru_cell_2/strided_slice_10?
gru_2/gru_cell_2/MatMul_5MatMulgru_2/gru_cell_2/mul_2:z:0*gru_2/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/MatMul_5?
'gru_2/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'gru_2/gru_cell_2/strided_slice_11/stack?
)gru_2/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)gru_2/gru_cell_2/strided_slice_11/stack_1?
)gru_2/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)gru_2/gru_cell_2/strided_slice_11/stack_2?
!gru_2/gru_cell_2/strided_slice_11StridedSlice!gru_2/gru_cell_2/unstack:output:10gru_2/gru_cell_2/strided_slice_11/stack:output:02gru_2/gru_cell_2/strided_slice_11/stack_1:output:02gru_2/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!gru_2/gru_cell_2/strided_slice_11?
gru_2/gru_cell_2/BiasAdd_5BiasAdd#gru_2/gru_cell_2/MatMul_5:product:0*gru_2/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/BiasAdd_5?
gru_2/gru_cell_2/mul_3Mulgru_2/gru_cell_2/Sigmoid_1:y:0#gru_2/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul_3?
gru_2/gru_cell_2/add_2AddV2#gru_2/gru_cell_2/BiasAdd_2:output:0gru_2/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/add_2?
gru_2/gru_cell_2/TanhTanhgru_2/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/Tanh?
gru_2/gru_cell_2/mul_4Mulgru_2/gru_cell_2/Sigmoid:y:0gru_2/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul_4u
gru_2/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_2/gru_cell_2/sub/x?
gru_2/gru_cell_2/subSubgru_2/gru_cell_2/sub/x:output:0gru_2/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/sub?
gru_2/gru_cell_2/mul_5Mulgru_2/gru_cell_2/sub:z:0gru_2/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/mul_5?
gru_2/gru_cell_2/add_3AddV2gru_2/gru_cell_2/mul_4:z:0gru_2/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_2/gru_cell_2/add_3?
#gru_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2%
#gru_2/TensorArrayV2_1/element_shape?
gru_2/TensorArrayV2_1TensorListReserve,gru_2/TensorArrayV2_1/element_shape:output:0gru_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_2/TensorArrayV2_1Z

gru_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_2/time?
gru_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
gru_2/while/maximum_iterationsv
gru_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_2/while/loop_counter?
gru_2/whileWhile!gru_2/while/loop_counter:output:0'gru_2/while/maximum_iterations:output:0gru_2/time:output:0gru_2/TensorArrayV2_1:handle:0gru_2/zeros:output:0gru_2/strided_slice_1:output:0=gru_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_2_gru_cell_2_readvariableop_resource*gru_2_gru_cell_2_readvariableop_1_resource*gru_2_gru_cell_2_readvariableop_4_resource*
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
gru_2_while_body_565217*#
condR
gru_2_while_cond_565216*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_2/while?
6gru_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   28
6gru_2/TensorArrayV2Stack/TensorListStack/element_shape?
(gru_2/TensorArrayV2Stack/TensorListStackTensorListStackgru_2/while:output:3?gru_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02*
(gru_2/TensorArrayV2Stack/TensorListStack?
gru_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_2/strided_slice_3/stack?
gru_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_2/strided_slice_3/stack_1?
gru_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_2/strided_slice_3/stack_2?
gru_2/strided_slice_3StridedSlice1gru_2/TensorArrayV2Stack/TensorListStack:tensor:0$gru_2/strided_slice_3/stack:output:0&gru_2/strided_slice_3/stack_1:output:0&gru_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_2/strided_slice_3?
gru_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_2/transpose_1/perm?
gru_2/transpose_1	Transpose1gru_2/TensorArrayV2Stack/TensorListStack:tensor:0gru_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_2/transpose_1r
gru_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_2/runtime?
layer_normalization_2/ShapeShapegru_2/strided_slice_3:output:0*
T0*
_output_shapes
:2
layer_normalization_2/Shape?
)layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_2/strided_slice/stack?
+layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_2/strided_slice/stack_1?
+layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_2/strided_slice/stack_2?
#layer_normalization_2/strided_sliceStridedSlice$layer_normalization_2/Shape:output:02layer_normalization_2/strided_slice/stack:output:04layer_normalization_2/strided_slice/stack_1:output:04layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_2/strided_slice|
layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_2/mul/x?
layer_normalization_2/mulMul$layer_normalization_2/mul/x:output:0,layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_2/mul?
+layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_2/strided_slice_1/stack?
-layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_2/strided_slice_1/stack_1?
-layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_2/strided_slice_1/stack_2?
%layer_normalization_2/strided_slice_1StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_1/stack:output:06layer_normalization_2/strided_slice_1/stack_1:output:06layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_2/strided_slice_1?
layer_normalization_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_2/mul_1/x?
layer_normalization_2/mul_1Mul&layer_normalization_2/mul_1/x:output:0.layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_2/mul_1?
%layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_2/Reshape/shape/0?
%layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_2/Reshape/shape/3?
#layer_normalization_2/Reshape/shapePack.layer_normalization_2/Reshape/shape/0:output:0layer_normalization_2/mul:z:0layer_normalization_2/mul_1:z:0.layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_2/Reshape/shape?
layer_normalization_2/ReshapeReshapegru_2/strided_slice_3:output:0,layer_normalization_2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_2/Reshape
layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_2/Const?
layer_normalization_2/Fill/dimsPacklayer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_2/Fill/dims?
layer_normalization_2/FillFill(layer_normalization_2/Fill/dims:output:0$layer_normalization_2/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_2/Fill?
layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_2/Const_1?
!layer_normalization_2/Fill_1/dimsPacklayer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_2/Fill_1/dims?
layer_normalization_2/Fill_1Fill*layer_normalization_2/Fill_1/dims:output:0&layer_normalization_2/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_2/Fill_1?
layer_normalization_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_2/Const_2?
layer_normalization_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_2/Const_3?
&layer_normalization_2/FusedBatchNormV3FusedBatchNormV3&layer_normalization_2/Reshape:output:0#layer_normalization_2/Fill:output:0%layer_normalization_2/Fill_1:output:0&layer_normalization_2/Const_2:output:0&layer_normalization_2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_2/FusedBatchNormV3?
layer_normalization_2/Reshape_1Reshape*layer_normalization_2/FusedBatchNormV3:y:0$layer_normalization_2/Shape:output:0*
T0*(
_output_shapes
:??????????2!
layer_normalization_2/Reshape_1?
*layer_normalization_2/mul_2/ReadVariableOpReadVariableOp3layer_normalization_2_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*layer_normalization_2/mul_2/ReadVariableOp?
layer_normalization_2/mul_2Mul(layer_normalization_2/Reshape_1:output:02layer_normalization_2/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_2/mul_2?
(layer_normalization_2/add/ReadVariableOpReadVariableOp1layer_normalization_2_add_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization_2/add/ReadVariableOp?
layer_normalization_2/addAddV2layer_normalization_2/mul_2:z:00layer_normalization_2/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_2/add?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMullayer_normalization_2/add:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd|
dense_12/SigmoidSigmoiddense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_12/Sigmoid?
IdentityIdentitydense_12/Sigmoid:y:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^gru_2/gru_cell_2/ReadVariableOp"^gru_2/gru_cell_2/ReadVariableOp_1"^gru_2/gru_cell_2/ReadVariableOp_2"^gru_2/gru_cell_2/ReadVariableOp_3"^gru_2/gru_cell_2/ReadVariableOp_4"^gru_2/gru_cell_2/ReadVariableOp_5"^gru_2/gru_cell_2/ReadVariableOp_6^gru_2/while)^layer_normalization_2/add/ReadVariableOp+^layer_normalization_2/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
gru_2/gru_cell_2/ReadVariableOpgru_2/gru_cell_2/ReadVariableOp2F
!gru_2/gru_cell_2/ReadVariableOp_1!gru_2/gru_cell_2/ReadVariableOp_12F
!gru_2/gru_cell_2/ReadVariableOp_2!gru_2/gru_cell_2/ReadVariableOp_22F
!gru_2/gru_cell_2/ReadVariableOp_3!gru_2/gru_cell_2/ReadVariableOp_32F
!gru_2/gru_cell_2/ReadVariableOp_4!gru_2/gru_cell_2/ReadVariableOp_42F
!gru_2/gru_cell_2/ReadVariableOp_5!gru_2/gru_cell_2/ReadVariableOp_52F
!gru_2/gru_cell_2/ReadVariableOp_6!gru_2/gru_cell_2/ReadVariableOp_62
gru_2/whilegru_2/while2T
(layer_normalization_2/add/ReadVariableOp(layer_normalization_2/add/ReadVariableOp2X
*layer_normalization_2/mul_2/ReadVariableOp*layer_normalization_2/mul_2/ReadVariableOp:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
while_cond_566841
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_566841___redundant_placeholder04
0while_while_cond_566841___redundant_placeholder14
0while_while_cond_566841___redundant_placeholder24
0while_while_cond_566841___redundant_placeholder3
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
?
?
C__inference_model_4_layer_call_and_return_conditional_losses_564958
input_5
gru_2_564940
gru_2_564942
gru_2_564944 
layer_normalization_2_564947 
layer_normalization_2_564949
dense_12_564952
dense_12_564954
identity?? dense_12/StatefulPartitionedCall?gru_2/StatefulPartitionedCall?-layer_normalization_2/StatefulPartitionedCall?
gru_2/StatefulPartitionedCallStatefulPartitionedCallinput_5gru_2_564940gru_2_564942gru_2_564944*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *J
fERC
A__inference_gru_2_layer_call_and_return_conditional_losses_5648212
gru_2/StatefulPartitionedCall?
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&gru_2/StatefulPartitionedCall:output:0layer_normalization_2_564947layer_normalization_2_564949*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_5648932/
-layer_normalization_2/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_12_564952dense_12_564954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5649202"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall^gru_2/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2>
gru_2/StatefulPartitionedCallgru_2/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
?"
?
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_567052

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
?
while_body_566842
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_2_readvariableop_resource_00
,while_gru_cell_2_readvariableop_1_resource_00
,while_gru_cell_2_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_2_readvariableop_resource.
*while_gru_cell_2_readvariableop_1_resource.
*while_gru_cell_2_readvariableop_4_resource??while/gru_cell_2/ReadVariableOp?!while/gru_cell_2/ReadVariableOp_1?!while/gru_cell_2/ReadVariableOp_2?!while/gru_cell_2/ReadVariableOp_3?!while/gru_cell_2/ReadVariableOp_4?!while/gru_cell_2/ReadVariableOp_5?!while/gru_cell_2/ReadVariableOp_6?
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
 while/gru_cell_2/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape?
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_2/ones_like/Const?
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/ones_like?
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_2/ReadVariableOp?
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_2/unstack?
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_1?
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack?
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice/stack_1?
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2?
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice?
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul?
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_2?
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_1/stack?
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_1/stack_1?
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2?
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1?
while/gru_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_1?
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_3?
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_2/strided_slice_2/stack?
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1?
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2?
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2?
while/gru_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_2?
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack?
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_3/stack_1?
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2?
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_3?
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd?
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_4/stack?
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_4/stack_1?
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2?
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_4?
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_1?
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_5/stack?
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1?
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2?
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_2/strided_slice_5?
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_2?
while/gru_cell_2/mulMulwhile_placeholder_2#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul?
while/gru_cell_2/mul_1Mulwhile_placeholder_2#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_1?
while/gru_cell_2/mul_2Mulwhile_placeholder_2#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_2?
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_4?
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack?
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_2/strided_slice_6/stack_1?
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2?
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6?
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_3?
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_5?
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_7/stack?
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_7/stack_1?
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2?
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7?
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_4?
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack?
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_8/stack_1?
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2?
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_8?
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_3?
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_9/stack?
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_9/stack_1?
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2?
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_9?
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_4?
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add?
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid?
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_1?
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid_1?
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_6?
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_2/strided_slice_10/stack?
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1?
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2?
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10?
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_2:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_5?
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_2/strided_slice_11/stack?
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1?
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2?
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_2/strided_slice_11?
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_5?
while/gru_cell_2/mul_3Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_3?
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_2?
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Tanh?
while/gru_cell_2/mul_4Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_4u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_2/sub/x?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/sub?
while/gru_cell_2/mul_5Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_5?
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_4:z:0while/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 
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
while_cond_564036
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_564036___redundant_placeholder04
0while_while_cond_564036___redundant_placeholder14
0while_while_cond_564036___redundant_placeholder24
0while_while_cond_564036___redundant_placeholder3
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
?
model_4_gru_2_while_body_5633398
4model_4_gru_2_while_model_4_gru_2_while_loop_counter>
:model_4_gru_2_while_model_4_gru_2_while_maximum_iterations#
model_4_gru_2_while_placeholder%
!model_4_gru_2_while_placeholder_1%
!model_4_gru_2_while_placeholder_27
3model_4_gru_2_while_model_4_gru_2_strided_slice_1_0s
omodel_4_gru_2_while_tensorarrayv2read_tensorlistgetitem_model_4_gru_2_tensorarrayunstack_tensorlistfromtensor_0<
8model_4_gru_2_while_gru_cell_2_readvariableop_resource_0>
:model_4_gru_2_while_gru_cell_2_readvariableop_1_resource_0>
:model_4_gru_2_while_gru_cell_2_readvariableop_4_resource_0 
model_4_gru_2_while_identity"
model_4_gru_2_while_identity_1"
model_4_gru_2_while_identity_2"
model_4_gru_2_while_identity_3"
model_4_gru_2_while_identity_45
1model_4_gru_2_while_model_4_gru_2_strided_slice_1q
mmodel_4_gru_2_while_tensorarrayv2read_tensorlistgetitem_model_4_gru_2_tensorarrayunstack_tensorlistfromtensor:
6model_4_gru_2_while_gru_cell_2_readvariableop_resource<
8model_4_gru_2_while_gru_cell_2_readvariableop_1_resource<
8model_4_gru_2_while_gru_cell_2_readvariableop_4_resource??-model_4/gru_2/while/gru_cell_2/ReadVariableOp?/model_4/gru_2/while/gru_cell_2/ReadVariableOp_1?/model_4/gru_2/while/gru_cell_2/ReadVariableOp_2?/model_4/gru_2/while/gru_cell_2/ReadVariableOp_3?/model_4/gru_2/while/gru_cell_2/ReadVariableOp_4?/model_4/gru_2/while/gru_cell_2/ReadVariableOp_5?/model_4/gru_2/while/gru_cell_2/ReadVariableOp_6?
Emodel_4/gru_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2G
Emodel_4/gru_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
7model_4/gru_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemomodel_4_gru_2_while_tensorarrayv2read_tensorlistgetitem_model_4_gru_2_tensorarrayunstack_tensorlistfromtensor_0model_4_gru_2_while_placeholderNmodel_4/gru_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype029
7model_4/gru_2/while/TensorArrayV2Read/TensorListGetItem?
.model_4/gru_2/while/gru_cell_2/ones_like/ShapeShape!model_4_gru_2_while_placeholder_2*
T0*
_output_shapes
:20
.model_4/gru_2/while/gru_cell_2/ones_like/Shape?
.model_4/gru_2/while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.model_4/gru_2/while/gru_cell_2/ones_like/Const?
(model_4/gru_2/while/gru_cell_2/ones_likeFill7model_4/gru_2/while/gru_cell_2/ones_like/Shape:output:07model_4/gru_2/while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2*
(model_4/gru_2/while/gru_cell_2/ones_like?
-model_4/gru_2/while/gru_cell_2/ReadVariableOpReadVariableOp8model_4_gru_2_while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-model_4/gru_2/while/gru_cell_2/ReadVariableOp?
&model_4/gru_2/while/gru_cell_2/unstackUnpack5model_4/gru_2/while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2(
&model_4/gru_2/while/gru_cell_2/unstack?
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_1ReadVariableOp:model_4_gru_2_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype021
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_1?
2model_4/gru_2/while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2model_4/gru_2/while/gru_cell_2/strided_slice/stack?
4model_4/gru_2/while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   26
4model_4/gru_2/while/gru_cell_2/strided_slice/stack_1?
4model_4/gru_2/while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model_4/gru_2/while/gru_cell_2/strided_slice/stack_2?
,model_4/gru_2/while/gru_cell_2/strided_sliceStridedSlice7model_4/gru_2/while/gru_cell_2/ReadVariableOp_1:value:0;model_4/gru_2/while/gru_cell_2/strided_slice/stack:output:0=model_4/gru_2/while/gru_cell_2/strided_slice/stack_1:output:0=model_4/gru_2/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2.
,model_4/gru_2/while/gru_cell_2/strided_slice?
%model_4/gru_2/while/gru_cell_2/MatMulMatMul>model_4/gru_2/while/TensorArrayV2Read/TensorListGetItem:item:05model_4/gru_2/while/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2'
%model_4/gru_2/while/gru_cell_2/MatMul?
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_2ReadVariableOp:model_4_gru_2_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype021
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_2?
4model_4/gru_2/while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   26
4model_4/gru_2/while/gru_cell_2/strided_slice_1/stack?
6model_4/gru_2/while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  28
6model_4/gru_2/while/gru_cell_2/strided_slice_1/stack_1?
6model_4/gru_2/while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6model_4/gru_2/while/gru_cell_2/strided_slice_1/stack_2?
.model_4/gru_2/while/gru_cell_2/strided_slice_1StridedSlice7model_4/gru_2/while/gru_cell_2/ReadVariableOp_2:value:0=model_4/gru_2/while/gru_cell_2/strided_slice_1/stack:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_1/stack_1:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask20
.model_4/gru_2/while/gru_cell_2/strided_slice_1?
'model_4/gru_2/while/gru_cell_2/MatMul_1MatMul>model_4/gru_2/while/TensorArrayV2Read/TensorListGetItem:item:07model_4/gru_2/while/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2)
'model_4/gru_2/while/gru_cell_2/MatMul_1?
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_3ReadVariableOp:model_4_gru_2_while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype021
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_3?
4model_4/gru_2/while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  26
4model_4/gru_2/while/gru_cell_2/strided_slice_2/stack?
6model_4/gru_2/while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        28
6model_4/gru_2/while/gru_cell_2/strided_slice_2/stack_1?
6model_4/gru_2/while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6model_4/gru_2/while/gru_cell_2/strided_slice_2/stack_2?
.model_4/gru_2/while/gru_cell_2/strided_slice_2StridedSlice7model_4/gru_2/while/gru_cell_2/ReadVariableOp_3:value:0=model_4/gru_2/while/gru_cell_2/strided_slice_2/stack:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_2/stack_1:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask20
.model_4/gru_2/while/gru_cell_2/strided_slice_2?
'model_4/gru_2/while/gru_cell_2/MatMul_2MatMul>model_4/gru_2/while/TensorArrayV2Read/TensorListGetItem:item:07model_4/gru_2/while/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2)
'model_4/gru_2/while/gru_cell_2/MatMul_2?
4model_4/gru_2/while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4model_4/gru_2/while/gru_cell_2/strided_slice_3/stack?
6model_4/gru_2/while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?28
6model_4/gru_2/while/gru_cell_2/strided_slice_3/stack_1?
6model_4/gru_2/while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model_4/gru_2/while/gru_cell_2/strided_slice_3/stack_2?
.model_4/gru_2/while/gru_cell_2/strided_slice_3StridedSlice/model_4/gru_2/while/gru_cell_2/unstack:output:0=model_4/gru_2/while/gru_cell_2/strided_slice_3/stack:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_3/stack_1:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask20
.model_4/gru_2/while/gru_cell_2/strided_slice_3?
&model_4/gru_2/while/gru_cell_2/BiasAddBiasAdd/model_4/gru_2/while/gru_cell_2/MatMul:product:07model_4/gru_2/while/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2(
&model_4/gru_2/while/gru_cell_2/BiasAdd?
4model_4/gru_2/while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?26
4model_4/gru_2/while/gru_cell_2/strided_slice_4/stack?
6model_4/gru_2/while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?28
6model_4/gru_2/while/gru_cell_2/strided_slice_4/stack_1?
6model_4/gru_2/while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model_4/gru_2/while/gru_cell_2/strided_slice_4/stack_2?
.model_4/gru_2/while/gru_cell_2/strided_slice_4StridedSlice/model_4/gru_2/while/gru_cell_2/unstack:output:0=model_4/gru_2/while/gru_cell_2/strided_slice_4/stack:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_4/stack_1:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?20
.model_4/gru_2/while/gru_cell_2/strided_slice_4?
(model_4/gru_2/while/gru_cell_2/BiasAdd_1BiasAdd1model_4/gru_2/while/gru_cell_2/MatMul_1:product:07model_4/gru_2/while/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2*
(model_4/gru_2/while/gru_cell_2/BiasAdd_1?
4model_4/gru_2/while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?26
4model_4/gru_2/while/gru_cell_2/strided_slice_5/stack?
6model_4/gru_2/while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6model_4/gru_2/while/gru_cell_2/strided_slice_5/stack_1?
6model_4/gru_2/while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model_4/gru_2/while/gru_cell_2/strided_slice_5/stack_2?
.model_4/gru_2/while/gru_cell_2/strided_slice_5StridedSlice/model_4/gru_2/while/gru_cell_2/unstack:output:0=model_4/gru_2/while/gru_cell_2/strided_slice_5/stack:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_5/stack_1:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask20
.model_4/gru_2/while/gru_cell_2/strided_slice_5?
(model_4/gru_2/while/gru_cell_2/BiasAdd_2BiasAdd1model_4/gru_2/while/gru_cell_2/MatMul_2:product:07model_4/gru_2/while/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2*
(model_4/gru_2/while/gru_cell_2/BiasAdd_2?
"model_4/gru_2/while/gru_cell_2/mulMul!model_4_gru_2_while_placeholder_21model_4/gru_2/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2$
"model_4/gru_2/while/gru_cell_2/mul?
$model_4/gru_2/while/gru_cell_2/mul_1Mul!model_4_gru_2_while_placeholder_21model_4/gru_2/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2&
$model_4/gru_2/while/gru_cell_2/mul_1?
$model_4/gru_2/while/gru_cell_2/mul_2Mul!model_4_gru_2_while_placeholder_21model_4/gru_2/while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2&
$model_4/gru_2/while/gru_cell_2/mul_2?
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_4ReadVariableOp:model_4_gru_2_while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype021
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_4?
4model_4/gru_2/while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4model_4/gru_2/while/gru_cell_2/strided_slice_6/stack?
6model_4/gru_2/while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   28
6model_4/gru_2/while/gru_cell_2/strided_slice_6/stack_1?
6model_4/gru_2/while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6model_4/gru_2/while/gru_cell_2/strided_slice_6/stack_2?
.model_4/gru_2/while/gru_cell_2/strided_slice_6StridedSlice7model_4/gru_2/while/gru_cell_2/ReadVariableOp_4:value:0=model_4/gru_2/while/gru_cell_2/strided_slice_6/stack:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_6/stack_1:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask20
.model_4/gru_2/while/gru_cell_2/strided_slice_6?
'model_4/gru_2/while/gru_cell_2/MatMul_3MatMul&model_4/gru_2/while/gru_cell_2/mul:z:07model_4/gru_2/while/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2)
'model_4/gru_2/while/gru_cell_2/MatMul_3?
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_5ReadVariableOp:model_4_gru_2_while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype021
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_5?
4model_4/gru_2/while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   26
4model_4/gru_2/while/gru_cell_2/strided_slice_7/stack?
6model_4/gru_2/while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  28
6model_4/gru_2/while/gru_cell_2/strided_slice_7/stack_1?
6model_4/gru_2/while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6model_4/gru_2/while/gru_cell_2/strided_slice_7/stack_2?
.model_4/gru_2/while/gru_cell_2/strided_slice_7StridedSlice7model_4/gru_2/while/gru_cell_2/ReadVariableOp_5:value:0=model_4/gru_2/while/gru_cell_2/strided_slice_7/stack:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_7/stack_1:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask20
.model_4/gru_2/while/gru_cell_2/strided_slice_7?
'model_4/gru_2/while/gru_cell_2/MatMul_4MatMul(model_4/gru_2/while/gru_cell_2/mul_1:z:07model_4/gru_2/while/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2)
'model_4/gru_2/while/gru_cell_2/MatMul_4?
4model_4/gru_2/while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4model_4/gru_2/while/gru_cell_2/strided_slice_8/stack?
6model_4/gru_2/while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?28
6model_4/gru_2/while/gru_cell_2/strided_slice_8/stack_1?
6model_4/gru_2/while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model_4/gru_2/while/gru_cell_2/strided_slice_8/stack_2?
.model_4/gru_2/while/gru_cell_2/strided_slice_8StridedSlice/model_4/gru_2/while/gru_cell_2/unstack:output:1=model_4/gru_2/while/gru_cell_2/strided_slice_8/stack:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_8/stack_1:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask20
.model_4/gru_2/while/gru_cell_2/strided_slice_8?
(model_4/gru_2/while/gru_cell_2/BiasAdd_3BiasAdd1model_4/gru_2/while/gru_cell_2/MatMul_3:product:07model_4/gru_2/while/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2*
(model_4/gru_2/while/gru_cell_2/BiasAdd_3?
4model_4/gru_2/while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?26
4model_4/gru_2/while/gru_cell_2/strided_slice_9/stack?
6model_4/gru_2/while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?28
6model_4/gru_2/while/gru_cell_2/strided_slice_9/stack_1?
6model_4/gru_2/while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model_4/gru_2/while/gru_cell_2/strided_slice_9/stack_2?
.model_4/gru_2/while/gru_cell_2/strided_slice_9StridedSlice/model_4/gru_2/while/gru_cell_2/unstack:output:1=model_4/gru_2/while/gru_cell_2/strided_slice_9/stack:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_9/stack_1:output:0?model_4/gru_2/while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?20
.model_4/gru_2/while/gru_cell_2/strided_slice_9?
(model_4/gru_2/while/gru_cell_2/BiasAdd_4BiasAdd1model_4/gru_2/while/gru_cell_2/MatMul_4:product:07model_4/gru_2/while/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2*
(model_4/gru_2/while/gru_cell_2/BiasAdd_4?
"model_4/gru_2/while/gru_cell_2/addAddV2/model_4/gru_2/while/gru_cell_2/BiasAdd:output:01model_4/gru_2/while/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2$
"model_4/gru_2/while/gru_cell_2/add?
&model_4/gru_2/while/gru_cell_2/SigmoidSigmoid&model_4/gru_2/while/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2(
&model_4/gru_2/while/gru_cell_2/Sigmoid?
$model_4/gru_2/while/gru_cell_2/add_1AddV21model_4/gru_2/while/gru_cell_2/BiasAdd_1:output:01model_4/gru_2/while/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2&
$model_4/gru_2/while/gru_cell_2/add_1?
(model_4/gru_2/while/gru_cell_2/Sigmoid_1Sigmoid(model_4/gru_2/while/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2*
(model_4/gru_2/while/gru_cell_2/Sigmoid_1?
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_6ReadVariableOp:model_4_gru_2_while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype021
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_6?
5model_4/gru_2/while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  27
5model_4/gru_2/while/gru_cell_2/strided_slice_10/stack?
7model_4/gru_2/while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7model_4/gru_2/while/gru_cell_2/strided_slice_10/stack_1?
7model_4/gru_2/while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7model_4/gru_2/while/gru_cell_2/strided_slice_10/stack_2?
/model_4/gru_2/while/gru_cell_2/strided_slice_10StridedSlice7model_4/gru_2/while/gru_cell_2/ReadVariableOp_6:value:0>model_4/gru_2/while/gru_cell_2/strided_slice_10/stack:output:0@model_4/gru_2/while/gru_cell_2/strided_slice_10/stack_1:output:0@model_4/gru_2/while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask21
/model_4/gru_2/while/gru_cell_2/strided_slice_10?
'model_4/gru_2/while/gru_cell_2/MatMul_5MatMul(model_4/gru_2/while/gru_cell_2/mul_2:z:08model_4/gru_2/while/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2)
'model_4/gru_2/while/gru_cell_2/MatMul_5?
5model_4/gru_2/while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?27
5model_4/gru_2/while/gru_cell_2/strided_slice_11/stack?
7model_4/gru_2/while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7model_4/gru_2/while/gru_cell_2/strided_slice_11/stack_1?
7model_4/gru_2/while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_4/gru_2/while/gru_cell_2/strided_slice_11/stack_2?
/model_4/gru_2/while/gru_cell_2/strided_slice_11StridedSlice/model_4/gru_2/while/gru_cell_2/unstack:output:1>model_4/gru_2/while/gru_cell_2/strided_slice_11/stack:output:0@model_4/gru_2/while/gru_cell_2/strided_slice_11/stack_1:output:0@model_4/gru_2/while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask21
/model_4/gru_2/while/gru_cell_2/strided_slice_11?
(model_4/gru_2/while/gru_cell_2/BiasAdd_5BiasAdd1model_4/gru_2/while/gru_cell_2/MatMul_5:product:08model_4/gru_2/while/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2*
(model_4/gru_2/while/gru_cell_2/BiasAdd_5?
$model_4/gru_2/while/gru_cell_2/mul_3Mul,model_4/gru_2/while/gru_cell_2/Sigmoid_1:y:01model_4/gru_2/while/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2&
$model_4/gru_2/while/gru_cell_2/mul_3?
$model_4/gru_2/while/gru_cell_2/add_2AddV21model_4/gru_2/while/gru_cell_2/BiasAdd_2:output:0(model_4/gru_2/while/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2&
$model_4/gru_2/while/gru_cell_2/add_2?
#model_4/gru_2/while/gru_cell_2/TanhTanh(model_4/gru_2/while/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2%
#model_4/gru_2/while/gru_cell_2/Tanh?
$model_4/gru_2/while/gru_cell_2/mul_4Mul*model_4/gru_2/while/gru_cell_2/Sigmoid:y:0!model_4_gru_2_while_placeholder_2*
T0*(
_output_shapes
:??????????2&
$model_4/gru_2/while/gru_cell_2/mul_4?
$model_4/gru_2/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$model_4/gru_2/while/gru_cell_2/sub/x?
"model_4/gru_2/while/gru_cell_2/subSub-model_4/gru_2/while/gru_cell_2/sub/x:output:0*model_4/gru_2/while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2$
"model_4/gru_2/while/gru_cell_2/sub?
$model_4/gru_2/while/gru_cell_2/mul_5Mul&model_4/gru_2/while/gru_cell_2/sub:z:0'model_4/gru_2/while/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2&
$model_4/gru_2/while/gru_cell_2/mul_5?
$model_4/gru_2/while/gru_cell_2/add_3AddV2(model_4/gru_2/while/gru_cell_2/mul_4:z:0(model_4/gru_2/while/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2&
$model_4/gru_2/while/gru_cell_2/add_3?
8model_4/gru_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!model_4_gru_2_while_placeholder_1model_4_gru_2_while_placeholder(model_4/gru_2/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype02:
8model_4/gru_2/while/TensorArrayV2Write/TensorListSetItemx
model_4/gru_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_4/gru_2/while/add/y?
model_4/gru_2/while/addAddV2model_4_gru_2_while_placeholder"model_4/gru_2/while/add/y:output:0*
T0*
_output_shapes
: 2
model_4/gru_2/while/add|
model_4/gru_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_4/gru_2/while/add_1/y?
model_4/gru_2/while/add_1AddV24model_4_gru_2_while_model_4_gru_2_while_loop_counter$model_4/gru_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_4/gru_2/while/add_1?
model_4/gru_2/while/IdentityIdentitymodel_4/gru_2/while/add_1:z:0.^model_4/gru_2/while/gru_cell_2/ReadVariableOp0^model_4/gru_2/while/gru_cell_2/ReadVariableOp_10^model_4/gru_2/while/gru_cell_2/ReadVariableOp_20^model_4/gru_2/while/gru_cell_2/ReadVariableOp_30^model_4/gru_2/while/gru_cell_2/ReadVariableOp_40^model_4/gru_2/while/gru_cell_2/ReadVariableOp_50^model_4/gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
model_4/gru_2/while/Identity?
model_4/gru_2/while/Identity_1Identity:model_4_gru_2_while_model_4_gru_2_while_maximum_iterations.^model_4/gru_2/while/gru_cell_2/ReadVariableOp0^model_4/gru_2/while/gru_cell_2/ReadVariableOp_10^model_4/gru_2/while/gru_cell_2/ReadVariableOp_20^model_4/gru_2/while/gru_cell_2/ReadVariableOp_30^model_4/gru_2/while/gru_cell_2/ReadVariableOp_40^model_4/gru_2/while/gru_cell_2/ReadVariableOp_50^model_4/gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2 
model_4/gru_2/while/Identity_1?
model_4/gru_2/while/Identity_2Identitymodel_4/gru_2/while/add:z:0.^model_4/gru_2/while/gru_cell_2/ReadVariableOp0^model_4/gru_2/while/gru_cell_2/ReadVariableOp_10^model_4/gru_2/while/gru_cell_2/ReadVariableOp_20^model_4/gru_2/while/gru_cell_2/ReadVariableOp_30^model_4/gru_2/while/gru_cell_2/ReadVariableOp_40^model_4/gru_2/while/gru_cell_2/ReadVariableOp_50^model_4/gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2 
model_4/gru_2/while/Identity_2?
model_4/gru_2/while/Identity_3IdentityHmodel_4/gru_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0.^model_4/gru_2/while/gru_cell_2/ReadVariableOp0^model_4/gru_2/while/gru_cell_2/ReadVariableOp_10^model_4/gru_2/while/gru_cell_2/ReadVariableOp_20^model_4/gru_2/while/gru_cell_2/ReadVariableOp_30^model_4/gru_2/while/gru_cell_2/ReadVariableOp_40^model_4/gru_2/while/gru_cell_2/ReadVariableOp_50^model_4/gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2 
model_4/gru_2/while/Identity_3?
model_4/gru_2/while/Identity_4Identity(model_4/gru_2/while/gru_cell_2/add_3:z:0.^model_4/gru_2/while/gru_cell_2/ReadVariableOp0^model_4/gru_2/while/gru_cell_2/ReadVariableOp_10^model_4/gru_2/while/gru_cell_2/ReadVariableOp_20^model_4/gru_2/while/gru_cell_2/ReadVariableOp_30^model_4/gru_2/while/gru_cell_2/ReadVariableOp_40^model_4/gru_2/while/gru_cell_2/ReadVariableOp_50^model_4/gru_2/while/gru_cell_2/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2 
model_4/gru_2/while/Identity_4"v
8model_4_gru_2_while_gru_cell_2_readvariableop_1_resource:model_4_gru_2_while_gru_cell_2_readvariableop_1_resource_0"v
8model_4_gru_2_while_gru_cell_2_readvariableop_4_resource:model_4_gru_2_while_gru_cell_2_readvariableop_4_resource_0"r
6model_4_gru_2_while_gru_cell_2_readvariableop_resource8model_4_gru_2_while_gru_cell_2_readvariableop_resource_0"E
model_4_gru_2_while_identity%model_4/gru_2/while/Identity:output:0"I
model_4_gru_2_while_identity_1'model_4/gru_2/while/Identity_1:output:0"I
model_4_gru_2_while_identity_2'model_4/gru_2/while/Identity_2:output:0"I
model_4_gru_2_while_identity_3'model_4/gru_2/while/Identity_3:output:0"I
model_4_gru_2_while_identity_4'model_4/gru_2/while/Identity_4:output:0"h
1model_4_gru_2_while_model_4_gru_2_strided_slice_13model_4_gru_2_while_model_4_gru_2_strided_slice_1_0"?
mmodel_4_gru_2_while_tensorarrayv2read_tensorlistgetitem_model_4_gru_2_tensorarrayunstack_tensorlistfromtensoromodel_4_gru_2_while_tensorarrayv2read_tensorlistgetitem_model_4_gru_2_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2^
-model_4/gru_2/while/gru_cell_2/ReadVariableOp-model_4/gru_2/while/gru_cell_2/ReadVariableOp2b
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_1/model_4/gru_2/while/gru_cell_2/ReadVariableOp_12b
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_2/model_4/gru_2/while/gru_cell_2/ReadVariableOp_22b
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_3/model_4/gru_2/while/gru_cell_2/ReadVariableOp_32b
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_4/model_4/gru_2/while/gru_cell_2/ReadVariableOp_42b
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_5/model_4/gru_2/while/gru_cell_2/ReadVariableOp_52b
/model_4/gru_2/while/gru_cell_2/ReadVariableOp_6/model_4/gru_2/while/gru_cell_2/ReadVariableOp_6: 
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
?J
?
__inference__traced_save_567447
file_prefix:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop6
2savev2_gru_2_gru_cell_2_kernel_read_readvariableop@
<savev2_gru_2_gru_cell_2_recurrent_kernel_read_readvariableop4
0savev2_gru_2_gru_cell_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableopB
>savev2_nadam_layer_normalization_2_gamma_m_read_readvariableopA
=savev2_nadam_layer_normalization_2_beta_m_read_readvariableop6
2savev2_nadam_dense_12_kernel_m_read_readvariableop4
0savev2_nadam_dense_12_bias_m_read_readvariableop>
:savev2_nadam_gru_2_gru_cell_2_kernel_m_read_readvariableopH
Dsavev2_nadam_gru_2_gru_cell_2_recurrent_kernel_m_read_readvariableop<
8savev2_nadam_gru_2_gru_cell_2_bias_m_read_readvariableopB
>savev2_nadam_layer_normalization_2_gamma_v_read_readvariableopA
=savev2_nadam_layer_normalization_2_beta_v_read_readvariableop6
2savev2_nadam_dense_12_kernel_v_read_readvariableop4
0savev2_nadam_dense_12_bias_v_read_readvariableop>
:savev2_nadam_gru_2_gru_cell_2_kernel_v_read_readvariableopH
Dsavev2_nadam_gru_2_gru_cell_2_recurrent_kernel_v_read_readvariableop<
8savev2_nadam_gru_2_gru_cell_2_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop2savev2_gru_2_gru_cell_2_kernel_read_readvariableop<savev2_gru_2_gru_cell_2_recurrent_kernel_read_readvariableop0savev2_gru_2_gru_cell_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop>savev2_nadam_layer_normalization_2_gamma_m_read_readvariableop=savev2_nadam_layer_normalization_2_beta_m_read_readvariableop2savev2_nadam_dense_12_kernel_m_read_readvariableop0savev2_nadam_dense_12_bias_m_read_readvariableop:savev2_nadam_gru_2_gru_cell_2_kernel_m_read_readvariableopDsavev2_nadam_gru_2_gru_cell_2_recurrent_kernel_m_read_readvariableop8savev2_nadam_gru_2_gru_cell_2_bias_m_read_readvariableop>savev2_nadam_layer_normalization_2_gamma_v_read_readvariableop=savev2_nadam_layer_normalization_2_beta_v_read_readvariableop2savev2_nadam_dense_12_kernel_v_read_readvariableop0savev2_nadam_dense_12_bias_v_read_readvariableop:savev2_nadam_gru_2_gru_cell_2_kernel_v_read_readvariableopDsavev2_nadam_gru_2_gru_cell_2_recurrent_kernel_v_read_readvariableop8savev2_nadam_gru_2_gru_cell_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?:	?:: : : : : : :		?:
??:	?: : :?:?:?:?:?:?:	?::		?:
??:	?:?:?:	?::		?:
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
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :%!

_output_shapes
:		?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:		?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:		?:& "
 
_output_shapes
:
??:%!!

_output_shapes
:	?:"

_output_shapes
: 
?
?
&__inference_gru_2_layer_call_fn_567010

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
*2
config_proto" 

CPU

GPU2*3,4J 8? *J
fERC
A__inference_gru_2_layer_call_and_return_conditional_losses_5648212
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
??
?
A__inference_gru_2_layer_call_and_return_conditional_losses_566376
inputs_0&
"gru_cell_2_readvariableop_resource(
$gru_cell_2_readvariableop_1_resource(
$gru_cell_2_readvariableop_4_resource
identity??gru_cell_2/ReadVariableOp?gru_cell_2/ReadVariableOp_1?gru_cell_2/ReadVariableOp_2?gru_cell_2/ReadVariableOp_3?gru_cell_2/ReadVariableOp_4?gru_cell_2/ReadVariableOp_5?gru_cell_2/ReadVariableOp_6?whileF
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
gru_cell_2/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/ones_like/Const?
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/ones_like?
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_2/ReadVariableOp?
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_2/unstack?
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_1?
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack?
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice/stack_1?
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2?
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice?
gru_cell_2/MatMulMatMulstrided_slice_2:output:0!gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul?
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_2?
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_1/stack?
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_1/stack_1?
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2?
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_1?
gru_cell_2/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_1?
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_3?
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_2/strided_slice_2/stack?
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1?
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2?
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_2?
gru_cell_2/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_2?
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack?
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_3/stack_1?
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2?
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_3?
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd?
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_4/stack?
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_4/stack_1?
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2?
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_4?
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_1?
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_5/stack?
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1?
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2?
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_5?
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_2?
gru_cell_2/mulMulzeros:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul?
gru_cell_2/mul_1Mulzeros:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_1?
gru_cell_2/mul_2Mulzeros:output:0gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_2?
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_4?
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack?
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_2/strided_slice_6/stack_1?
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2?
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_6?
gru_cell_2/MatMul_3MatMulgru_cell_2/mul:z:0#gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_3?
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_5?
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_7/stack?
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_7/stack_1?
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2?
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_7?
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_4?
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack?
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_8/stack_1?
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2?
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_8?
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_3?
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_9/stack?
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_9/stack_1?
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2?
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_9?
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_4?
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/addz
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid?
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_1?
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid_1?
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_6?
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_2/strided_slice_10/stack?
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1?
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2?
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_10?
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_2:z:0$gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_5?
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_2/strided_slice_11/stack?
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1?
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2?
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_11?
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_5?
gru_cell_2/mul_3Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_3?
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_2s
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Tanh?
gru_cell_2/mul_4Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_4i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/sub/x?
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/sub?
gru_cell_2/mul_5Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_5?
gru_cell_2/add_3AddV2gru_cell_2/mul_4:z:0gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
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
while_body_566230*
condR
while_cond_566229*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:^ Z
4
_output_shapes"
 :??????????????????	
"
_user_specified_name
inputs/0
܍
?
"__inference__traced_restore_567556
file_prefix0
,assignvariableop_layer_normalization_2_gamma1
-assignvariableop_1_layer_normalization_2_beta&
"assignvariableop_2_dense_12_kernel$
 assignvariableop_3_dense_12_bias!
assignvariableop_4_nadam_iter#
assignvariableop_5_nadam_beta_1#
assignvariableop_6_nadam_beta_2"
assignvariableop_7_nadam_decay*
&assignvariableop_8_nadam_learning_rate+
'assignvariableop_9_nadam_momentum_cache/
+assignvariableop_10_gru_2_gru_cell_2_kernel9
5assignvariableop_11_gru_2_gru_cell_2_recurrent_kernel-
)assignvariableop_12_gru_2_gru_cell_2_bias
assignvariableop_13_total
assignvariableop_14_count&
"assignvariableop_15_true_positives&
"assignvariableop_16_true_negatives'
#assignvariableop_17_false_positives'
#assignvariableop_18_false_negatives;
7assignvariableop_19_nadam_layer_normalization_2_gamma_m:
6assignvariableop_20_nadam_layer_normalization_2_beta_m/
+assignvariableop_21_nadam_dense_12_kernel_m-
)assignvariableop_22_nadam_dense_12_bias_m7
3assignvariableop_23_nadam_gru_2_gru_cell_2_kernel_mA
=assignvariableop_24_nadam_gru_2_gru_cell_2_recurrent_kernel_m5
1assignvariableop_25_nadam_gru_2_gru_cell_2_bias_m;
7assignvariableop_26_nadam_layer_normalization_2_gamma_v:
6assignvariableop_27_nadam_layer_normalization_2_beta_v/
+assignvariableop_28_nadam_dense_12_kernel_v-
)assignvariableop_29_nadam_dense_12_bias_v7
3assignvariableop_30_nadam_gru_2_gru_cell_2_kernel_vA
=assignvariableop_31_nadam_gru_2_gru_cell_2_recurrent_kernel_v5
1assignvariableop_32_nadam_gru_2_gru_cell_2_bias_v
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp,assignvariableop_layer_normalization_2_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp-assignvariableop_1_layer_normalization_2_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_12_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_12_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_nadam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_nadam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_nadam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_nadam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_nadam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp'assignvariableop_9_nadam_momentum_cacheIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_gru_2_gru_cell_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp5assignvariableop_11_gru_2_gru_cell_2_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_gru_2_gru_cell_2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_true_negativesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_false_positivesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_false_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp7assignvariableop_19_nadam_layer_normalization_2_gamma_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp6assignvariableop_20_nadam_layer_normalization_2_beta_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_nadam_dense_12_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_nadam_dense_12_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp3assignvariableop_23_nadam_gru_2_gru_cell_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp=assignvariableop_24_nadam_gru_2_gru_cell_2_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp1assignvariableop_25_nadam_gru_2_gru_cell_2_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp7assignvariableop_26_nadam_layer_normalization_2_gamma_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp6assignvariableop_27_nadam_layer_normalization_2_beta_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_nadam_dense_12_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_nadam_dense_12_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp3assignvariableop_30_nadam_gru_2_gru_cell_2_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp=assignvariableop_31_nadam_gru_2_gru_cell_2_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp1assignvariableop_32_nadam_gru_2_gru_cell_2_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33?
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
&__inference_gru_2_layer_call_fn_566398
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
*2
config_proto" 

CPU

GPU2*3,4J 8? *J
fERC
A__inference_gru_2_layer_call_and_return_conditional_losses_5642192
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
?
while_body_564675
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_2_readvariableop_resource_00
,while_gru_cell_2_readvariableop_1_resource_00
,while_gru_cell_2_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_2_readvariableop_resource.
*while_gru_cell_2_readvariableop_1_resource.
*while_gru_cell_2_readvariableop_4_resource??while/gru_cell_2/ReadVariableOp?!while/gru_cell_2/ReadVariableOp_1?!while/gru_cell_2/ReadVariableOp_2?!while/gru_cell_2/ReadVariableOp_3?!while/gru_cell_2/ReadVariableOp_4?!while/gru_cell_2/ReadVariableOp_5?!while/gru_cell_2/ReadVariableOp_6?
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
 while/gru_cell_2/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape?
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_2/ones_like/Const?
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/ones_like?
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_2/ReadVariableOp?
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_2/unstack?
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_1?
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack?
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice/stack_1?
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2?
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice?
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul?
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_2?
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_1/stack?
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_1/stack_1?
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2?
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1?
while/gru_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_1?
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_3?
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_2/strided_slice_2/stack?
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1?
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2?
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2?
while/gru_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_2?
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack?
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_3/stack_1?
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2?
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_3?
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd?
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_4/stack?
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_4/stack_1?
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2?
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_4?
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_1?
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_5/stack?
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1?
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2?
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_2/strided_slice_5?
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_2?
while/gru_cell_2/mulMulwhile_placeholder_2#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul?
while/gru_cell_2/mul_1Mulwhile_placeholder_2#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_1?
while/gru_cell_2/mul_2Mulwhile_placeholder_2#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_2?
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_4?
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack?
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_2/strided_slice_6/stack_1?
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2?
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6?
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_3?
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_5?
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_7/stack?
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_7/stack_1?
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2?
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7?
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_4?
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack?
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_8/stack_1?
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2?
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_8?
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_3?
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_9/stack?
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_9/stack_1?
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2?
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_9?
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_4?
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add?
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid?
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_1?
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid_1?
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_6?
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_2/strided_slice_10/stack?
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1?
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2?
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10?
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_2:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_5?
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_2/strided_slice_11/stack?
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1?
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2?
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_2/strided_slice_11?
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_5?
while/gru_cell_2/mul_3Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_3?
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_2?
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Tanh?
while/gru_cell_2/mul_4Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_4u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_2/sub/x?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/sub?
while/gru_cell_2/mul_5Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_5?
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_4:z:0while/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 
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
+__inference_gru_cell_2_layer_call_fn_567325

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
*2
config_proto" 

CPU

GPU2*3,4J 8? *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_5637782
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
??
?
A__inference_gru_2_layer_call_and_return_conditional_losses_566105
inputs_0&
"gru_cell_2_readvariableop_resource(
$gru_cell_2_readvariableop_1_resource(
$gru_cell_2_readvariableop_4_resource
identity??gru_cell_2/ReadVariableOp?gru_cell_2/ReadVariableOp_1?gru_cell_2/ReadVariableOp_2?gru_cell_2/ReadVariableOp_3?gru_cell_2/ReadVariableOp_4?gru_cell_2/ReadVariableOp_5?gru_cell_2/ReadVariableOp_6?whileF
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
gru_cell_2/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell_2/ones_like/Shape}
gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/ones_like/Const?
gru_cell_2/ones_likeFill#gru_cell_2/ones_like/Shape:output:0#gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/ones_likey
gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout/Const?
gru_cell_2/dropout/MulMulgru_cell_2/ones_like:output:0!gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout/Mul?
gru_cell_2/dropout/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout/Shape?
/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform!gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??721
/gru_cell_2/dropout/random_uniform/RandomUniform?
!gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell_2/dropout/GreaterEqual/y?
gru_cell_2/dropout/GreaterEqualGreaterEqual8gru_cell_2/dropout/random_uniform/RandomUniform:output:0*gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell_2/dropout/GreaterEqual?
gru_cell_2/dropout/CastCast#gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_2/dropout/Cast?
gru_cell_2/dropout/Mul_1Mulgru_cell_2/dropout/Mul:z:0gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout/Mul_1}
gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_1/Const?
gru_cell_2/dropout_1/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_1/Mul?
gru_cell_2/dropout_1/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_1/Shape?
1gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???23
1gru_cell_2/dropout_1/random_uniform/RandomUniform?
#gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_2/dropout_1/GreaterEqual/y?
!gru_cell_2/dropout_1/GreaterEqualGreaterEqual:gru_cell_2/dropout_1/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_2/dropout_1/GreaterEqual?
gru_cell_2/dropout_1/CastCast%gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_2/dropout_1/Cast?
gru_cell_2/dropout_1/Mul_1Mulgru_cell_2/dropout_1/Mul:z:0gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_1/Mul_1}
gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell_2/dropout_2/Const?
gru_cell_2/dropout_2/MulMulgru_cell_2/ones_like:output:0#gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_2/Mul?
gru_cell_2/dropout_2/ShapeShapegru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell_2/dropout_2/Shape?
1gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform#gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??Q23
1gru_cell_2/dropout_2/random_uniform/RandomUniform?
#gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru_cell_2/dropout_2/GreaterEqual/y?
!gru_cell_2/dropout_2/GreaterEqualGreaterEqual:gru_cell_2/dropout_2/random_uniform/RandomUniform:output:0,gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru_cell_2/dropout_2/GreaterEqual?
gru_cell_2/dropout_2/CastCast%gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell_2/dropout_2/Cast?
gru_cell_2/dropout_2/Mul_1Mulgru_cell_2/dropout_2/Mul:z:0gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/dropout_2/Mul_1?
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_2/ReadVariableOp?
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_2/unstack?
gru_cell_2/ReadVariableOp_1ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_1?
gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell_2/strided_slice/stack?
 gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice/stack_1?
 gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell_2/strided_slice/stack_2?
gru_cell_2/strided_sliceStridedSlice#gru_cell_2/ReadVariableOp_1:value:0'gru_cell_2/strided_slice/stack:output:0)gru_cell_2/strided_slice/stack_1:output:0)gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice?
gru_cell_2/MatMulMatMulstrided_slice_2:output:0!gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul?
gru_cell_2/ReadVariableOp_2ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_2?
 gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_1/stack?
"gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_1/stack_1?
"gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_1/stack_2?
gru_cell_2/strided_slice_1StridedSlice#gru_cell_2/ReadVariableOp_2:value:0)gru_cell_2/strided_slice_1/stack:output:0+gru_cell_2/strided_slice_1/stack_1:output:0+gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_1?
gru_cell_2/MatMul_1MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_1?
gru_cell_2/ReadVariableOp_3ReadVariableOp$gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell_2/ReadVariableOp_3?
 gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell_2/strided_slice_2/stack?
"gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"gru_cell_2/strided_slice_2/stack_1?
"gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_2/stack_2?
gru_cell_2/strided_slice_2StridedSlice#gru_cell_2/ReadVariableOp_3:value:0)gru_cell_2/strided_slice_2/stack:output:0+gru_cell_2/strided_slice_2/stack_1:output:0+gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell_2/strided_slice_2?
gru_cell_2/MatMul_2MatMulstrided_slice_2:output:0#gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_2?
 gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_3/stack?
"gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_3/stack_1?
"gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_3/stack_2?
gru_cell_2/strided_slice_3StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_3/stack:output:0+gru_cell_2/strided_slice_3/stack_1:output:0+gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_3?
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0#gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd?
 gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_4/stack?
"gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_4/stack_1?
"gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_4/stack_2?
gru_cell_2/strided_slice_4StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_4/stack:output:0+gru_cell_2/strided_slice_4/stack_1:output:0+gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_4?
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0#gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_1?
 gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_5/stack?
"gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gru_cell_2/strided_slice_5/stack_1?
"gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_5/stack_2?
gru_cell_2/strided_slice_5StridedSlicegru_cell_2/unstack:output:0)gru_cell_2/strided_slice_5/stack:output:0+gru_cell_2/strided_slice_5/stack_1:output:0+gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_5?
gru_cell_2/BiasAdd_2BiasAddgru_cell_2/MatMul_2:product:0#gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_2?
gru_cell_2/mulMulzeros:output:0gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul?
gru_cell_2/mul_1Mulzeros:output:0gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_1?
gru_cell_2/mul_2Mulzeros:output:0gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_2?
gru_cell_2/ReadVariableOp_4ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_4?
 gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell_2/strided_slice_6/stack?
"gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru_cell_2/strided_slice_6/stack_1?
"gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_6/stack_2?
gru_cell_2/strided_slice_6StridedSlice#gru_cell_2/ReadVariableOp_4:value:0)gru_cell_2/strided_slice_6/stack:output:0+gru_cell_2/strided_slice_6/stack_1:output:0+gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_6?
gru_cell_2/MatMul_3MatMulgru_cell_2/mul:z:0#gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_3?
gru_cell_2/ReadVariableOp_5ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_5?
 gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell_2/strided_slice_7/stack?
"gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru_cell_2/strided_slice_7/stack_1?
"gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru_cell_2/strided_slice_7/stack_2?
gru_cell_2/strided_slice_7StridedSlice#gru_cell_2/ReadVariableOp_5:value:0)gru_cell_2/strided_slice_7/stack:output:0+gru_cell_2/strided_slice_7/stack_1:output:0+gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_7?
gru_cell_2/MatMul_4MatMulgru_cell_2/mul_1:z:0#gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_4?
 gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell_2/strided_slice_8/stack?
"gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_8/stack_1?
"gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_8/stack_2?
gru_cell_2/strided_slice_8StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_8/stack:output:0+gru_cell_2/strided_slice_8/stack_1:output:0+gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell_2/strided_slice_8?
gru_cell_2/BiasAdd_3BiasAddgru_cell_2/MatMul_3:product:0#gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_3?
 gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell_2/strided_slice_9/stack?
"gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2$
"gru_cell_2/strided_slice_9/stack_1?
"gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gru_cell_2/strided_slice_9/stack_2?
gru_cell_2/strided_slice_9StridedSlicegru_cell_2/unstack:output:1)gru_cell_2/strided_slice_9/stack:output:0+gru_cell_2/strided_slice_9/stack_1:output:0+gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell_2/strided_slice_9?
gru_cell_2/BiasAdd_4BiasAddgru_cell_2/MatMul_4:product:0#gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_4?
gru_cell_2/addAddV2gru_cell_2/BiasAdd:output:0gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/addz
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid?
gru_cell_2/add_1AddV2gru_cell_2/BiasAdd_1:output:0gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_1?
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Sigmoid_1?
gru_cell_2/ReadVariableOp_6ReadVariableOp$gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell_2/ReadVariableOp_6?
!gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!gru_cell_2/strided_slice_10/stack?
#gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#gru_cell_2/strided_slice_10/stack_1?
#gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#gru_cell_2/strided_slice_10/stack_2?
gru_cell_2/strided_slice_10StridedSlice#gru_cell_2/ReadVariableOp_6:value:0*gru_cell_2/strided_slice_10/stack:output:0,gru_cell_2/strided_slice_10/stack_1:output:0,gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell_2/strided_slice_10?
gru_cell_2/MatMul_5MatMulgru_cell_2/mul_2:z:0$gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/MatMul_5?
!gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2#
!gru_cell_2/strided_slice_11/stack?
#gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#gru_cell_2/strided_slice_11/stack_1?
#gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#gru_cell_2/strided_slice_11/stack_2?
gru_cell_2/strided_slice_11StridedSlicegru_cell_2/unstack:output:1*gru_cell_2/strided_slice_11/stack:output:0,gru_cell_2/strided_slice_11/stack_1:output:0,gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell_2/strided_slice_11?
gru_cell_2/BiasAdd_5BiasAddgru_cell_2/MatMul_5:product:0$gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/BiasAdd_5?
gru_cell_2/mul_3Mulgru_cell_2/Sigmoid_1:y:0gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_3?
gru_cell_2/add_2AddV2gru_cell_2/BiasAdd_2:output:0gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_2s
gru_cell_2/TanhTanhgru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/Tanh?
gru_cell_2/mul_4Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_4i
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_2/sub/x?
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/sub?
gru_cell_2/mul_5Mulgru_cell_2/sub:z:0gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/mul_5?
gru_cell_2/add_3AddV2gru_cell_2/mul_4:z:0gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_2/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource$gru_cell_2_readvariableop_1_resource$gru_cell_2_readvariableop_4_resource*
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
while_body_565935*
condR
while_cond_565934*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell_2/ReadVariableOp^gru_cell_2/ReadVariableOp_1^gru_cell_2/ReadVariableOp_2^gru_cell_2/ReadVariableOp_3^gru_cell_2/ReadVariableOp_4^gru_cell_2/ReadVariableOp_5^gru_cell_2/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2:
gru_cell_2/ReadVariableOp_1gru_cell_2/ReadVariableOp_12:
gru_cell_2/ReadVariableOp_2gru_cell_2/ReadVariableOp_22:
gru_cell_2/ReadVariableOp_3gru_cell_2/ReadVariableOp_32:
gru_cell_2/ReadVariableOp_4gru_cell_2/ReadVariableOp_42:
gru_cell_2/ReadVariableOp_5gru_cell_2/ReadVariableOp_52:
gru_cell_2/ReadVariableOp_6gru_cell_2/ReadVariableOp_62
whilewhile:^ Z
4
_output_shapes"
 :??????????????????	
"
_user_specified_name
inputs/0
?
?
C__inference_model_4_layer_call_and_return_conditional_losses_564937
input_5
gru_2_564844
gru_2_564846
gru_2_564848 
layer_normalization_2_564904 
layer_normalization_2_564906
dense_12_564931
dense_12_564933
identity?? dense_12/StatefulPartitionedCall?gru_2/StatefulPartitionedCall?-layer_normalization_2/StatefulPartitionedCall?
gru_2/StatefulPartitionedCallStatefulPartitionedCallinput_5gru_2_564844gru_2_564846gru_2_564848*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *J
fERC
A__inference_gru_2_layer_call_and_return_conditional_losses_5645502
gru_2/StatefulPartitionedCall?
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&gru_2/StatefulPartitionedCall:output:0layer_normalization_2_564904layer_normalization_2_564906*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_5648932/
-layer_normalization_2/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_12_564931dense_12_564933*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5649202"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall^gru_2/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2>
gru_2/StatefulPartitionedCallgru_2/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
?!
?
while_body_564155
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_2_564177_0
while_gru_cell_2_564179_0
while_gru_cell_2_564181_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_2_564177
while_gru_cell_2_564179
while_gru_cell_2_564181??(while/gru_cell_2/StatefulPartitionedCall?
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
(while/gru_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_2_564177_0while_gru_cell_2_564179_0while_gru_cell_2_564181_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_5637782*
(while/gru_cell_2/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_2/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity1while/gru_cell_2/StatefulPartitionedCall:output:1)^while/gru_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4"4
while_gru_cell_2_564177while_gru_cell_2_564177_0"4
while_gru_cell_2_564179while_gru_cell_2_564179_0"4
while_gru_cell_2_564181while_gru_cell_2_564181_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2T
(while/gru_cell_2/StatefulPartitionedCall(while/gru_cell_2/StatefulPartitionedCall: 
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
?<
?
A__inference_gru_2_layer_call_and_return_conditional_losses_564101

inputs
gru_cell_2_564025
gru_cell_2_564027
gru_cell_2_564029
identity??"gru_cell_2/StatefulPartitionedCall?whileD
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
"gru_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_2_564025gru_cell_2_564027gru_cell_2_564029*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*3,4J 8? *O
fJRH
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_5636822$
"gru_cell_2/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_2_564025gru_cell_2_564027gru_cell_2_564029*
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
while_body_564037*
condR
while_cond_564036*9
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
IdentityIdentitystrided_slice_3:output:0#^gru_cell_2/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::2H
"gru_cell_2/StatefulPartitionedCall"gru_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????	
 
_user_specified_nameinputs
?
?
while_cond_564154
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_564154___redundant_placeholder04
0while_while_cond_564154___redundant_placeholder14
0while_while_cond_564154___redundant_placeholder24
0while_while_cond_564154___redundant_placeholder3
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
Ū
?
!__inference__wrapped_model_563530
input_54
0model_4_gru_2_gru_cell_2_readvariableop_resource6
2model_4_gru_2_gru_cell_2_readvariableop_1_resource6
2model_4_gru_2_gru_cell_2_readvariableop_4_resource?
;model_4_layer_normalization_2_mul_2_readvariableop_resource=
9model_4_layer_normalization_2_add_readvariableop_resource3
/model_4_dense_12_matmul_readvariableop_resource4
0model_4_dense_12_biasadd_readvariableop_resource
identity??'model_4/dense_12/BiasAdd/ReadVariableOp?&model_4/dense_12/MatMul/ReadVariableOp?'model_4/gru_2/gru_cell_2/ReadVariableOp?)model_4/gru_2/gru_cell_2/ReadVariableOp_1?)model_4/gru_2/gru_cell_2/ReadVariableOp_2?)model_4/gru_2/gru_cell_2/ReadVariableOp_3?)model_4/gru_2/gru_cell_2/ReadVariableOp_4?)model_4/gru_2/gru_cell_2/ReadVariableOp_5?)model_4/gru_2/gru_cell_2/ReadVariableOp_6?model_4/gru_2/while?0model_4/layer_normalization_2/add/ReadVariableOp?2model_4/layer_normalization_2/mul_2/ReadVariableOpa
model_4/gru_2/ShapeShapeinput_5*
T0*
_output_shapes
:2
model_4/gru_2/Shape?
!model_4/gru_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model_4/gru_2/strided_slice/stack?
#model_4/gru_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#model_4/gru_2/strided_slice/stack_1?
#model_4/gru_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model_4/gru_2/strided_slice/stack_2?
model_4/gru_2/strided_sliceStridedSlicemodel_4/gru_2/Shape:output:0*model_4/gru_2/strided_slice/stack:output:0,model_4/gru_2/strided_slice/stack_1:output:0,model_4/gru_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_4/gru_2/strided_slicey
model_4/gru_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_4/gru_2/zeros/mul/y?
model_4/gru_2/zeros/mulMul$model_4/gru_2/strided_slice:output:0"model_4/gru_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_4/gru_2/zeros/mul{
model_4/gru_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_4/gru_2/zeros/Less/y?
model_4/gru_2/zeros/LessLessmodel_4/gru_2/zeros/mul:z:0#model_4/gru_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_4/gru_2/zeros/Less
model_4/gru_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
model_4/gru_2/zeros/packed/1?
model_4/gru_2/zeros/packedPack$model_4/gru_2/strided_slice:output:0%model_4/gru_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_4/gru_2/zeros/packed{
model_4/gru_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_4/gru_2/zeros/Const?
model_4/gru_2/zerosFill#model_4/gru_2/zeros/packed:output:0"model_4/gru_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
model_4/gru_2/zeros?
model_4/gru_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model_4/gru_2/transpose/perm?
model_4/gru_2/transpose	Transposeinput_5%model_4/gru_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????	2
model_4/gru_2/transposey
model_4/gru_2/Shape_1Shapemodel_4/gru_2/transpose:y:0*
T0*
_output_shapes
:2
model_4/gru_2/Shape_1?
#model_4/gru_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_4/gru_2/strided_slice_1/stack?
%model_4/gru_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_4/gru_2/strided_slice_1/stack_1?
%model_4/gru_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_4/gru_2/strided_slice_1/stack_2?
model_4/gru_2/strided_slice_1StridedSlicemodel_4/gru_2/Shape_1:output:0,model_4/gru_2/strided_slice_1/stack:output:0.model_4/gru_2/strided_slice_1/stack_1:output:0.model_4/gru_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_4/gru_2/strided_slice_1?
)model_4/gru_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)model_4/gru_2/TensorArrayV2/element_shape?
model_4/gru_2/TensorArrayV2TensorListReserve2model_4/gru_2/TensorArrayV2/element_shape:output:0&model_4/gru_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_4/gru_2/TensorArrayV2?
Cmodel_4/gru_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2E
Cmodel_4/gru_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
5model_4/gru_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_4/gru_2/transpose:y:0Lmodel_4/gru_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5model_4/gru_2/TensorArrayUnstack/TensorListFromTensor?
#model_4/gru_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_4/gru_2/strided_slice_2/stack?
%model_4/gru_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_4/gru_2/strided_slice_2/stack_1?
%model_4/gru_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_4/gru_2/strided_slice_2/stack_2?
model_4/gru_2/strided_slice_2StridedSlicemodel_4/gru_2/transpose:y:0,model_4/gru_2/strided_slice_2/stack:output:0.model_4/gru_2/strided_slice_2/stack_1:output:0.model_4/gru_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
model_4/gru_2/strided_slice_2?
(model_4/gru_2/gru_cell_2/ones_like/ShapeShapemodel_4/gru_2/zeros:output:0*
T0*
_output_shapes
:2*
(model_4/gru_2/gru_cell_2/ones_like/Shape?
(model_4/gru_2/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(model_4/gru_2/gru_cell_2/ones_like/Const?
"model_4/gru_2/gru_cell_2/ones_likeFill1model_4/gru_2/gru_cell_2/ones_like/Shape:output:01model_4/gru_2/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2$
"model_4/gru_2/gru_cell_2/ones_like?
'model_4/gru_2/gru_cell_2/ReadVariableOpReadVariableOp0model_4_gru_2_gru_cell_2_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'model_4/gru_2/gru_cell_2/ReadVariableOp?
 model_4/gru_2/gru_cell_2/unstackUnpack/model_4/gru_2/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 model_4/gru_2/gru_cell_2/unstack?
)model_4/gru_2/gru_cell_2/ReadVariableOp_1ReadVariableOp2model_4_gru_2_gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02+
)model_4/gru_2/gru_cell_2/ReadVariableOp_1?
,model_4/gru_2/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,model_4/gru_2/gru_cell_2/strided_slice/stack?
.model_4/gru_2/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   20
.model_4/gru_2/gru_cell_2/strided_slice/stack_1?
.model_4/gru_2/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/gru_2/gru_cell_2/strided_slice/stack_2?
&model_4/gru_2/gru_cell_2/strided_sliceStridedSlice1model_4/gru_2/gru_cell_2/ReadVariableOp_1:value:05model_4/gru_2/gru_cell_2/strided_slice/stack:output:07model_4/gru_2/gru_cell_2/strided_slice/stack_1:output:07model_4/gru_2/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&model_4/gru_2/gru_cell_2/strided_slice?
model_4/gru_2/gru_cell_2/MatMulMatMul&model_4/gru_2/strided_slice_2:output:0/model_4/gru_2/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2!
model_4/gru_2/gru_cell_2/MatMul?
)model_4/gru_2/gru_cell_2/ReadVariableOp_2ReadVariableOp2model_4_gru_2_gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02+
)model_4/gru_2/gru_cell_2/ReadVariableOp_2?
.model_4/gru_2/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   20
.model_4/gru_2/gru_cell_2/strided_slice_1/stack?
0model_4/gru_2/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  22
0model_4/gru_2/gru_cell_2/strided_slice_1/stack_1?
0model_4/gru_2/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model_4/gru_2/gru_cell_2/strided_slice_1/stack_2?
(model_4/gru_2/gru_cell_2/strided_slice_1StridedSlice1model_4/gru_2/gru_cell_2/ReadVariableOp_2:value:07model_4/gru_2/gru_cell_2/strided_slice_1/stack:output:09model_4/gru_2/gru_cell_2/strided_slice_1/stack_1:output:09model_4/gru_2/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2*
(model_4/gru_2/gru_cell_2/strided_slice_1?
!model_4/gru_2/gru_cell_2/MatMul_1MatMul&model_4/gru_2/strided_slice_2:output:01model_4/gru_2/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2#
!model_4/gru_2/gru_cell_2/MatMul_1?
)model_4/gru_2/gru_cell_2/ReadVariableOp_3ReadVariableOp2model_4_gru_2_gru_cell_2_readvariableop_1_resource*
_output_shapes
:		?*
dtype02+
)model_4/gru_2/gru_cell_2/ReadVariableOp_3?
.model_4/gru_2/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  20
.model_4/gru_2/gru_cell_2/strided_slice_2/stack?
0model_4/gru_2/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0model_4/gru_2/gru_cell_2/strided_slice_2/stack_1?
0model_4/gru_2/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model_4/gru_2/gru_cell_2/strided_slice_2/stack_2?
(model_4/gru_2/gru_cell_2/strided_slice_2StridedSlice1model_4/gru_2/gru_cell_2/ReadVariableOp_3:value:07model_4/gru_2/gru_cell_2/strided_slice_2/stack:output:09model_4/gru_2/gru_cell_2/strided_slice_2/stack_1:output:09model_4/gru_2/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2*
(model_4/gru_2/gru_cell_2/strided_slice_2?
!model_4/gru_2/gru_cell_2/MatMul_2MatMul&model_4/gru_2/strided_slice_2:output:01model_4/gru_2/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2#
!model_4/gru_2/gru_cell_2/MatMul_2?
.model_4/gru_2/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_4/gru_2/gru_cell_2/strided_slice_3/stack?
0model_4/gru_2/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model_4/gru_2/gru_cell_2/strided_slice_3/stack_1?
0model_4/gru_2/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_4/gru_2/gru_cell_2/strided_slice_3/stack_2?
(model_4/gru_2/gru_cell_2/strided_slice_3StridedSlice)model_4/gru_2/gru_cell_2/unstack:output:07model_4/gru_2/gru_cell_2/strided_slice_3/stack:output:09model_4/gru_2/gru_cell_2/strided_slice_3/stack_1:output:09model_4/gru_2/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2*
(model_4/gru_2/gru_cell_2/strided_slice_3?
 model_4/gru_2/gru_cell_2/BiasAddBiasAdd)model_4/gru_2/gru_cell_2/MatMul:product:01model_4/gru_2/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2"
 model_4/gru_2/gru_cell_2/BiasAdd?
.model_4/gru_2/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?20
.model_4/gru_2/gru_cell_2/strided_slice_4/stack?
0model_4/gru_2/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model_4/gru_2/gru_cell_2/strided_slice_4/stack_1?
0model_4/gru_2/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_4/gru_2/gru_cell_2/strided_slice_4/stack_2?
(model_4/gru_2/gru_cell_2/strided_slice_4StridedSlice)model_4/gru_2/gru_cell_2/unstack:output:07model_4/gru_2/gru_cell_2/strided_slice_4/stack:output:09model_4/gru_2/gru_cell_2/strided_slice_4/stack_1:output:09model_4/gru_2/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2*
(model_4/gru_2/gru_cell_2/strided_slice_4?
"model_4/gru_2/gru_cell_2/BiasAdd_1BiasAdd+model_4/gru_2/gru_cell_2/MatMul_1:product:01model_4/gru_2/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2$
"model_4/gru_2/gru_cell_2/BiasAdd_1?
.model_4/gru_2/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?20
.model_4/gru_2/gru_cell_2/strided_slice_5/stack?
0model_4/gru_2/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 22
0model_4/gru_2/gru_cell_2/strided_slice_5/stack_1?
0model_4/gru_2/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_4/gru_2/gru_cell_2/strided_slice_5/stack_2?
(model_4/gru_2/gru_cell_2/strided_slice_5StridedSlice)model_4/gru_2/gru_cell_2/unstack:output:07model_4/gru_2/gru_cell_2/strided_slice_5/stack:output:09model_4/gru_2/gru_cell_2/strided_slice_5/stack_1:output:09model_4/gru_2/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2*
(model_4/gru_2/gru_cell_2/strided_slice_5?
"model_4/gru_2/gru_cell_2/BiasAdd_2BiasAdd+model_4/gru_2/gru_cell_2/MatMul_2:product:01model_4/gru_2/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2$
"model_4/gru_2/gru_cell_2/BiasAdd_2?
model_4/gru_2/gru_cell_2/mulMulmodel_4/gru_2/zeros:output:0+model_4/gru_2/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
model_4/gru_2/gru_cell_2/mul?
model_4/gru_2/gru_cell_2/mul_1Mulmodel_4/gru_2/zeros:output:0+model_4/gru_2/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2 
model_4/gru_2/gru_cell_2/mul_1?
model_4/gru_2/gru_cell_2/mul_2Mulmodel_4/gru_2/zeros:output:0+model_4/gru_2/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2 
model_4/gru_2/gru_cell_2/mul_2?
)model_4/gru_2/gru_cell_2/ReadVariableOp_4ReadVariableOp2model_4_gru_2_gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02+
)model_4/gru_2/gru_cell_2/ReadVariableOp_4?
.model_4/gru_2/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.model_4/gru_2/gru_cell_2/strided_slice_6/stack?
0model_4/gru_2/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0model_4/gru_2/gru_cell_2/strided_slice_6/stack_1?
0model_4/gru_2/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model_4/gru_2/gru_cell_2/strided_slice_6/stack_2?
(model_4/gru_2/gru_cell_2/strided_slice_6StridedSlice1model_4/gru_2/gru_cell_2/ReadVariableOp_4:value:07model_4/gru_2/gru_cell_2/strided_slice_6/stack:output:09model_4/gru_2/gru_cell_2/strided_slice_6/stack_1:output:09model_4/gru_2/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(model_4/gru_2/gru_cell_2/strided_slice_6?
!model_4/gru_2/gru_cell_2/MatMul_3MatMul model_4/gru_2/gru_cell_2/mul:z:01model_4/gru_2/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2#
!model_4/gru_2/gru_cell_2/MatMul_3?
)model_4/gru_2/gru_cell_2/ReadVariableOp_5ReadVariableOp2model_4_gru_2_gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02+
)model_4/gru_2/gru_cell_2/ReadVariableOp_5?
.model_4/gru_2/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   20
.model_4/gru_2/gru_cell_2/strided_slice_7/stack?
0model_4/gru_2/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  22
0model_4/gru_2/gru_cell_2/strided_slice_7/stack_1?
0model_4/gru_2/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model_4/gru_2/gru_cell_2/strided_slice_7/stack_2?
(model_4/gru_2/gru_cell_2/strided_slice_7StridedSlice1model_4/gru_2/gru_cell_2/ReadVariableOp_5:value:07model_4/gru_2/gru_cell_2/strided_slice_7/stack:output:09model_4/gru_2/gru_cell_2/strided_slice_7/stack_1:output:09model_4/gru_2/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(model_4/gru_2/gru_cell_2/strided_slice_7?
!model_4/gru_2/gru_cell_2/MatMul_4MatMul"model_4/gru_2/gru_cell_2/mul_1:z:01model_4/gru_2/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2#
!model_4/gru_2/gru_cell_2/MatMul_4?
.model_4/gru_2/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_4/gru_2/gru_cell_2/strided_slice_8/stack?
0model_4/gru_2/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model_4/gru_2/gru_cell_2/strided_slice_8/stack_1?
0model_4/gru_2/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_4/gru_2/gru_cell_2/strided_slice_8/stack_2?
(model_4/gru_2/gru_cell_2/strided_slice_8StridedSlice)model_4/gru_2/gru_cell_2/unstack:output:17model_4/gru_2/gru_cell_2/strided_slice_8/stack:output:09model_4/gru_2/gru_cell_2/strided_slice_8/stack_1:output:09model_4/gru_2/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2*
(model_4/gru_2/gru_cell_2/strided_slice_8?
"model_4/gru_2/gru_cell_2/BiasAdd_3BiasAdd+model_4/gru_2/gru_cell_2/MatMul_3:product:01model_4/gru_2/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2$
"model_4/gru_2/gru_cell_2/BiasAdd_3?
.model_4/gru_2/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?20
.model_4/gru_2/gru_cell_2/strided_slice_9/stack?
0model_4/gru_2/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model_4/gru_2/gru_cell_2/strided_slice_9/stack_1?
0model_4/gru_2/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_4/gru_2/gru_cell_2/strided_slice_9/stack_2?
(model_4/gru_2/gru_cell_2/strided_slice_9StridedSlice)model_4/gru_2/gru_cell_2/unstack:output:17model_4/gru_2/gru_cell_2/strided_slice_9/stack:output:09model_4/gru_2/gru_cell_2/strided_slice_9/stack_1:output:09model_4/gru_2/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2*
(model_4/gru_2/gru_cell_2/strided_slice_9?
"model_4/gru_2/gru_cell_2/BiasAdd_4BiasAdd+model_4/gru_2/gru_cell_2/MatMul_4:product:01model_4/gru_2/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2$
"model_4/gru_2/gru_cell_2/BiasAdd_4?
model_4/gru_2/gru_cell_2/addAddV2)model_4/gru_2/gru_cell_2/BiasAdd:output:0+model_4/gru_2/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
model_4/gru_2/gru_cell_2/add?
 model_4/gru_2/gru_cell_2/SigmoidSigmoid model_4/gru_2/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2"
 model_4/gru_2/gru_cell_2/Sigmoid?
model_4/gru_2/gru_cell_2/add_1AddV2+model_4/gru_2/gru_cell_2/BiasAdd_1:output:0+model_4/gru_2/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2 
model_4/gru_2/gru_cell_2/add_1?
"model_4/gru_2/gru_cell_2/Sigmoid_1Sigmoid"model_4/gru_2/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"model_4/gru_2/gru_cell_2/Sigmoid_1?
)model_4/gru_2/gru_cell_2/ReadVariableOp_6ReadVariableOp2model_4_gru_2_gru_cell_2_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02+
)model_4/gru_2/gru_cell_2/ReadVariableOp_6?
/model_4/gru_2/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  21
/model_4/gru_2/gru_cell_2/strided_slice_10/stack?
1model_4/gru_2/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1model_4/gru_2/gru_cell_2/strided_slice_10/stack_1?
1model_4/gru_2/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1model_4/gru_2/gru_cell_2/strided_slice_10/stack_2?
)model_4/gru_2/gru_cell_2/strided_slice_10StridedSlice1model_4/gru_2/gru_cell_2/ReadVariableOp_6:value:08model_4/gru_2/gru_cell_2/strided_slice_10/stack:output:0:model_4/gru_2/gru_cell_2/strided_slice_10/stack_1:output:0:model_4/gru_2/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2+
)model_4/gru_2/gru_cell_2/strided_slice_10?
!model_4/gru_2/gru_cell_2/MatMul_5MatMul"model_4/gru_2/gru_cell_2/mul_2:z:02model_4/gru_2/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2#
!model_4/gru_2/gru_cell_2/MatMul_5?
/model_4/gru_2/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?21
/model_4/gru_2/gru_cell_2/strided_slice_11/stack?
1model_4/gru_2/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1model_4/gru_2/gru_cell_2/strided_slice_11/stack_1?
1model_4/gru_2/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_4/gru_2/gru_cell_2/strided_slice_11/stack_2?
)model_4/gru_2/gru_cell_2/strided_slice_11StridedSlice)model_4/gru_2/gru_cell_2/unstack:output:18model_4/gru_2/gru_cell_2/strided_slice_11/stack:output:0:model_4/gru_2/gru_cell_2/strided_slice_11/stack_1:output:0:model_4/gru_2/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2+
)model_4/gru_2/gru_cell_2/strided_slice_11?
"model_4/gru_2/gru_cell_2/BiasAdd_5BiasAdd+model_4/gru_2/gru_cell_2/MatMul_5:product:02model_4/gru_2/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2$
"model_4/gru_2/gru_cell_2/BiasAdd_5?
model_4/gru_2/gru_cell_2/mul_3Mul&model_4/gru_2/gru_cell_2/Sigmoid_1:y:0+model_4/gru_2/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2 
model_4/gru_2/gru_cell_2/mul_3?
model_4/gru_2/gru_cell_2/add_2AddV2+model_4/gru_2/gru_cell_2/BiasAdd_2:output:0"model_4/gru_2/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2 
model_4/gru_2/gru_cell_2/add_2?
model_4/gru_2/gru_cell_2/TanhTanh"model_4/gru_2/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
model_4/gru_2/gru_cell_2/Tanh?
model_4/gru_2/gru_cell_2/mul_4Mul$model_4/gru_2/gru_cell_2/Sigmoid:y:0model_4/gru_2/zeros:output:0*
T0*(
_output_shapes
:??????????2 
model_4/gru_2/gru_cell_2/mul_4?
model_4/gru_2/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
model_4/gru_2/gru_cell_2/sub/x?
model_4/gru_2/gru_cell_2/subSub'model_4/gru_2/gru_cell_2/sub/x:output:0$model_4/gru_2/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
model_4/gru_2/gru_cell_2/sub?
model_4/gru_2/gru_cell_2/mul_5Mul model_4/gru_2/gru_cell_2/sub:z:0!model_4/gru_2/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2 
model_4/gru_2/gru_cell_2/mul_5?
model_4/gru_2/gru_cell_2/add_3AddV2"model_4/gru_2/gru_cell_2/mul_4:z:0"model_4/gru_2/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2 
model_4/gru_2/gru_cell_2/add_3?
+model_4/gru_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2-
+model_4/gru_2/TensorArrayV2_1/element_shape?
model_4/gru_2/TensorArrayV2_1TensorListReserve4model_4/gru_2/TensorArrayV2_1/element_shape:output:0&model_4/gru_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_4/gru_2/TensorArrayV2_1j
model_4/gru_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_4/gru_2/time?
&model_4/gru_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&model_4/gru_2/while/maximum_iterations?
 model_4/gru_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 model_4/gru_2/while/loop_counter?
model_4/gru_2/whileWhile)model_4/gru_2/while/loop_counter:output:0/model_4/gru_2/while/maximum_iterations:output:0model_4/gru_2/time:output:0&model_4/gru_2/TensorArrayV2_1:handle:0model_4/gru_2/zeros:output:0&model_4/gru_2/strided_slice_1:output:0Emodel_4/gru_2/TensorArrayUnstack/TensorListFromTensor:output_handle:00model_4_gru_2_gru_cell_2_readvariableop_resource2model_4_gru_2_gru_cell_2_readvariableop_1_resource2model_4_gru_2_gru_cell_2_readvariableop_4_resource*
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
model_4_gru_2_while_body_563339*+
cond#R!
model_4_gru_2_while_cond_563338*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
model_4/gru_2/while?
>model_4/gru_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>model_4/gru_2/TensorArrayV2Stack/TensorListStack/element_shape?
0model_4/gru_2/TensorArrayV2Stack/TensorListStackTensorListStackmodel_4/gru_2/while:output:3Gmodel_4/gru_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype022
0model_4/gru_2/TensorArrayV2Stack/TensorListStack?
#model_4/gru_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#model_4/gru_2/strided_slice_3/stack?
%model_4/gru_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_4/gru_2/strided_slice_3/stack_1?
%model_4/gru_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_4/gru_2/strided_slice_3/stack_2?
model_4/gru_2/strided_slice_3StridedSlice9model_4/gru_2/TensorArrayV2Stack/TensorListStack:tensor:0,model_4/gru_2/strided_slice_3/stack:output:0.model_4/gru_2/strided_slice_3/stack_1:output:0.model_4/gru_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
model_4/gru_2/strided_slice_3?
model_4/gru_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model_4/gru_2/transpose_1/perm?
model_4/gru_2/transpose_1	Transpose9model_4/gru_2/TensorArrayV2Stack/TensorListStack:tensor:0'model_4/gru_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
model_4/gru_2/transpose_1?
model_4/gru_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_4/gru_2/runtime?
#model_4/layer_normalization_2/ShapeShape&model_4/gru_2/strided_slice_3:output:0*
T0*
_output_shapes
:2%
#model_4/layer_normalization_2/Shape?
1model_4/layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_4/layer_normalization_2/strided_slice/stack?
3model_4/layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_4/layer_normalization_2/strided_slice/stack_1?
3model_4/layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_4/layer_normalization_2/strided_slice/stack_2?
+model_4/layer_normalization_2/strided_sliceStridedSlice,model_4/layer_normalization_2/Shape:output:0:model_4/layer_normalization_2/strided_slice/stack:output:0<model_4/layer_normalization_2/strided_slice/stack_1:output:0<model_4/layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_4/layer_normalization_2/strided_slice?
#model_4/layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_4/layer_normalization_2/mul/x?
!model_4/layer_normalization_2/mulMul,model_4/layer_normalization_2/mul/x:output:04model_4/layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: 2#
!model_4/layer_normalization_2/mul?
3model_4/layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3model_4/layer_normalization_2/strided_slice_1/stack?
5model_4/layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5model_4/layer_normalization_2/strided_slice_1/stack_1?
5model_4/layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5model_4/layer_normalization_2/strided_slice_1/stack_2?
-model_4/layer_normalization_2/strided_slice_1StridedSlice,model_4/layer_normalization_2/Shape:output:0<model_4/layer_normalization_2/strided_slice_1/stack:output:0>model_4/layer_normalization_2/strided_slice_1/stack_1:output:0>model_4/layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-model_4/layer_normalization_2/strided_slice_1?
%model_4/layer_normalization_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_4/layer_normalization_2/mul_1/x?
#model_4/layer_normalization_2/mul_1Mul.model_4/layer_normalization_2/mul_1/x:output:06model_4/layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: 2%
#model_4/layer_normalization_2/mul_1?
-model_4/layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_4/layer_normalization_2/Reshape/shape/0?
-model_4/layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_4/layer_normalization_2/Reshape/shape/3?
+model_4/layer_normalization_2/Reshape/shapePack6model_4/layer_normalization_2/Reshape/shape/0:output:0%model_4/layer_normalization_2/mul:z:0'model_4/layer_normalization_2/mul_1:z:06model_4/layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+model_4/layer_normalization_2/Reshape/shape?
%model_4/layer_normalization_2/ReshapeReshape&model_4/gru_2/strided_slice_3:output:04model_4/layer_normalization_2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%model_4/layer_normalization_2/Reshape?
#model_4/layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#model_4/layer_normalization_2/Const?
'model_4/layer_normalization_2/Fill/dimsPack%model_4/layer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:2)
'model_4/layer_normalization_2/Fill/dims?
"model_4/layer_normalization_2/FillFill0model_4/layer_normalization_2/Fill/dims:output:0,model_4/layer_normalization_2/Const:output:0*
T0*#
_output_shapes
:?????????2$
"model_4/layer_normalization_2/Fill?
%model_4/layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2'
%model_4/layer_normalization_2/Const_1?
)model_4/layer_normalization_2/Fill_1/dimsPack%model_4/layer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:2+
)model_4/layer_normalization_2/Fill_1/dims?
$model_4/layer_normalization_2/Fill_1Fill2model_4/layer_normalization_2/Fill_1/dims:output:0.model_4/layer_normalization_2/Const_1:output:0*
T0*#
_output_shapes
:?????????2&
$model_4/layer_normalization_2/Fill_1?
%model_4/layer_normalization_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2'
%model_4/layer_normalization_2/Const_2?
%model_4/layer_normalization_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2'
%model_4/layer_normalization_2/Const_3?
.model_4/layer_normalization_2/FusedBatchNormV3FusedBatchNormV3.model_4/layer_normalization_2/Reshape:output:0+model_4/layer_normalization_2/Fill:output:0-model_4/layer_normalization_2/Fill_1:output:0.model_4/layer_normalization_2/Const_2:output:0.model_4/layer_normalization_2/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:20
.model_4/layer_normalization_2/FusedBatchNormV3?
'model_4/layer_normalization_2/Reshape_1Reshape2model_4/layer_normalization_2/FusedBatchNormV3:y:0,model_4/layer_normalization_2/Shape:output:0*
T0*(
_output_shapes
:??????????2)
'model_4/layer_normalization_2/Reshape_1?
2model_4/layer_normalization_2/mul_2/ReadVariableOpReadVariableOp;model_4_layer_normalization_2_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_4/layer_normalization_2/mul_2/ReadVariableOp?
#model_4/layer_normalization_2/mul_2Mul0model_4/layer_normalization_2/Reshape_1:output:0:model_4/layer_normalization_2/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#model_4/layer_normalization_2/mul_2?
0model_4/layer_normalization_2/add/ReadVariableOpReadVariableOp9model_4_layer_normalization_2_add_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_4/layer_normalization_2/add/ReadVariableOp?
!model_4/layer_normalization_2/addAddV2'model_4/layer_normalization_2/mul_2:z:08model_4/layer_normalization_2/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!model_4/layer_normalization_2/add?
&model_4/dense_12/MatMul/ReadVariableOpReadVariableOp/model_4_dense_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_4/dense_12/MatMul/ReadVariableOp?
model_4/dense_12/MatMulMatMul%model_4/layer_normalization_2/add:z:0.model_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/dense_12/MatMul?
'model_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_4/dense_12/BiasAdd/ReadVariableOp?
model_4/dense_12/BiasAddBiasAdd!model_4/dense_12/MatMul:product:0/model_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/dense_12/BiasAdd?
model_4/dense_12/SigmoidSigmoid!model_4/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/dense_12/Sigmoid?
IdentityIdentitymodel_4/dense_12/Sigmoid:y:0(^model_4/dense_12/BiasAdd/ReadVariableOp'^model_4/dense_12/MatMul/ReadVariableOp(^model_4/gru_2/gru_cell_2/ReadVariableOp*^model_4/gru_2/gru_cell_2/ReadVariableOp_1*^model_4/gru_2/gru_cell_2/ReadVariableOp_2*^model_4/gru_2/gru_cell_2/ReadVariableOp_3*^model_4/gru_2/gru_cell_2/ReadVariableOp_4*^model_4/gru_2/gru_cell_2/ReadVariableOp_5*^model_4/gru_2/gru_cell_2/ReadVariableOp_6^model_4/gru_2/while1^model_4/layer_normalization_2/add/ReadVariableOp3^model_4/layer_normalization_2/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2R
'model_4/dense_12/BiasAdd/ReadVariableOp'model_4/dense_12/BiasAdd/ReadVariableOp2P
&model_4/dense_12/MatMul/ReadVariableOp&model_4/dense_12/MatMul/ReadVariableOp2R
'model_4/gru_2/gru_cell_2/ReadVariableOp'model_4/gru_2/gru_cell_2/ReadVariableOp2V
)model_4/gru_2/gru_cell_2/ReadVariableOp_1)model_4/gru_2/gru_cell_2/ReadVariableOp_12V
)model_4/gru_2/gru_cell_2/ReadVariableOp_2)model_4/gru_2/gru_cell_2/ReadVariableOp_22V
)model_4/gru_2/gru_cell_2/ReadVariableOp_3)model_4/gru_2/gru_cell_2/ReadVariableOp_32V
)model_4/gru_2/gru_cell_2/ReadVariableOp_4)model_4/gru_2/gru_cell_2/ReadVariableOp_42V
)model_4/gru_2/gru_cell_2/ReadVariableOp_5)model_4/gru_2/gru_cell_2/ReadVariableOp_52V
)model_4/gru_2/gru_cell_2/ReadVariableOp_6)model_4/gru_2/gru_cell_2/ReadVariableOp_62*
model_4/gru_2/whilemodel_4/gru_2/while2d
0model_4/layer_normalization_2/add/ReadVariableOp0model_4/layer_normalization_2/add/ReadVariableOp2h
2model_4/layer_normalization_2/mul_2/ReadVariableOp2model_4/layer_normalization_2/mul_2/ReadVariableOp:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
??
?
while_body_566547
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_2_readvariableop_resource_00
,while_gru_cell_2_readvariableop_1_resource_00
,while_gru_cell_2_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_2_readvariableop_resource.
*while_gru_cell_2_readvariableop_1_resource.
*while_gru_cell_2_readvariableop_4_resource??while/gru_cell_2/ReadVariableOp?!while/gru_cell_2/ReadVariableOp_1?!while/gru_cell_2/ReadVariableOp_2?!while/gru_cell_2/ReadVariableOp_3?!while/gru_cell_2/ReadVariableOp_4?!while/gru_cell_2/ReadVariableOp_5?!while/gru_cell_2/ReadVariableOp_6?
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
 while/gru_cell_2/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape?
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_2/ones_like/Const?
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/ones_like?
while/gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell_2/dropout/Const?
while/gru_cell_2/dropout/MulMul#while/gru_cell_2/ones_like:output:0'while/gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/dropout/Mul?
while/gru_cell_2/dropout/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell_2/dropout/Shape?
5while/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell_2/dropout/random_uniform/RandomUniform?
'while/gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell_2/dropout/GreaterEqual/y?
%while/gru_cell_2/dropout/GreaterEqualGreaterEqual>while/gru_cell_2/dropout/random_uniform/RandomUniform:output:00while/gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell_2/dropout/GreaterEqual?
while/gru_cell_2/dropout/CastCast)while/gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell_2/dropout/Cast?
while/gru_cell_2/dropout/Mul_1Mul while/gru_cell_2/dropout/Mul:z:0!while/gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_2/dropout/Mul_1?
 while/gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_1/Const?
while/gru_cell_2/dropout_1/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_2/dropout_1/Mul?
 while/gru_cell_2/dropout_1/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_1/Shape?
7while/gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??w29
7while/gru_cell_2/dropout_1/random_uniform/RandomUniform?
)while/gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_2/dropout_1/GreaterEqual/y?
'while/gru_cell_2/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_2/dropout_1/GreaterEqual?
while/gru_cell_2/dropout_1/CastCast+while/gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_2/dropout_1/Cast?
 while/gru_cell_2/dropout_1/Mul_1Mul"while/gru_cell_2/dropout_1/Mul:z:0#while/gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_2/dropout_1/Mul_1?
 while/gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_2/Const?
while/gru_cell_2/dropout_2/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_2/dropout_2/Mul?
 while/gru_cell_2/dropout_2/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_2/Shape?
7while/gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??S29
7while/gru_cell_2/dropout_2/random_uniform/RandomUniform?
)while/gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_2/dropout_2/GreaterEqual/y?
'while/gru_cell_2/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_2/dropout_2/GreaterEqual?
while/gru_cell_2/dropout_2/CastCast+while/gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_2/dropout_2/Cast?
 while/gru_cell_2/dropout_2/Mul_1Mul"while/gru_cell_2/dropout_2/Mul:z:0#while/gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_2/dropout_2/Mul_1?
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_2/ReadVariableOp?
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_2/unstack?
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_1?
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack?
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice/stack_1?
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2?
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice?
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul?
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_2?
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_1/stack?
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_1/stack_1?
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2?
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1?
while/gru_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_1?
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_3?
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_2/strided_slice_2/stack?
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1?
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2?
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2?
while/gru_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_2?
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack?
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_3/stack_1?
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2?
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_3?
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd?
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_4/stack?
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_4/stack_1?
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2?
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_4?
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_1?
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_5/stack?
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1?
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2?
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_2/strided_slice_5?
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_2?
while/gru_cell_2/mulMulwhile_placeholder_2"while/gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul?
while/gru_cell_2/mul_1Mulwhile_placeholder_2$while/gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_1?
while/gru_cell_2/mul_2Mulwhile_placeholder_2$while/gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_2?
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_4?
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack?
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_2/strided_slice_6/stack_1?
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2?
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6?
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_3?
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_5?
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_7/stack?
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_7/stack_1?
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2?
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7?
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_4?
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack?
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_8/stack_1?
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2?
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_8?
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_3?
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_9/stack?
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_9/stack_1?
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2?
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_9?
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_4?
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add?
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid?
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_1?
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid_1?
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_6?
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_2/strided_slice_10/stack?
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1?
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2?
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10?
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_2:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_5?
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_2/strided_slice_11/stack?
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1?
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2?
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_2/strided_slice_11?
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_5?
while/gru_cell_2/mul_3Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_3?
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_2?
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Tanh?
while/gru_cell_2/mul_4Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_4u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_2/sub/x?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/sub?
while/gru_cell_2/mul_5Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_5?
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_4:z:0while/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 
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
while_cond_564674
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_564674___redundant_placeholder04
0while_while_cond_564674___redundant_placeholder14
0while_while_cond_564674___redundant_placeholder24
0while_while_cond_564674___redundant_placeholder3
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
while_body_565935
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_2_readvariableop_resource_00
,while_gru_cell_2_readvariableop_1_resource_00
,while_gru_cell_2_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_2_readvariableop_resource.
*while_gru_cell_2_readvariableop_1_resource.
*while_gru_cell_2_readvariableop_4_resource??while/gru_cell_2/ReadVariableOp?!while/gru_cell_2/ReadVariableOp_1?!while/gru_cell_2/ReadVariableOp_2?!while/gru_cell_2/ReadVariableOp_3?!while/gru_cell_2/ReadVariableOp_4?!while/gru_cell_2/ReadVariableOp_5?!while/gru_cell_2/ReadVariableOp_6?
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
 while/gru_cell_2/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape?
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_2/ones_like/Const?
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/ones_like?
while/gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell_2/dropout/Const?
while/gru_cell_2/dropout/MulMul#while/gru_cell_2/ones_like:output:0'while/gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/dropout/Mul?
while/gru_cell_2/dropout/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell_2/dropout/Shape?
5while/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??27
5while/gru_cell_2/dropout/random_uniform/RandomUniform?
'while/gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell_2/dropout/GreaterEqual/y?
%while/gru_cell_2/dropout/GreaterEqualGreaterEqual>while/gru_cell_2/dropout/random_uniform/RandomUniform:output:00while/gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell_2/dropout/GreaterEqual?
while/gru_cell_2/dropout/CastCast)while/gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell_2/dropout/Cast?
while/gru_cell_2/dropout/Mul_1Mul while/gru_cell_2/dropout/Mul:z:0!while/gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_2/dropout/Mul_1?
 while/gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_1/Const?
while/gru_cell_2/dropout_1/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_2/dropout_1/Mul?
 while/gru_cell_2/dropout_1/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_1/Shape?
7while/gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???29
7while/gru_cell_2/dropout_1/random_uniform/RandomUniform?
)while/gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_2/dropout_1/GreaterEqual/y?
'while/gru_cell_2/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_2/dropout_1/GreaterEqual?
while/gru_cell_2/dropout_1/CastCast+while/gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_2/dropout_1/Cast?
 while/gru_cell_2/dropout_1/Mul_1Mul"while/gru_cell_2/dropout_1/Mul:z:0#while/gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_2/dropout_1/Mul_1?
 while/gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_2/Const?
while/gru_cell_2/dropout_2/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_2/dropout_2/Mul?
 while/gru_cell_2/dropout_2/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_2/Shape?
7while/gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??529
7while/gru_cell_2/dropout_2/random_uniform/RandomUniform?
)while/gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_2/dropout_2/GreaterEqual/y?
'while/gru_cell_2/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_2/dropout_2/GreaterEqual?
while/gru_cell_2/dropout_2/CastCast+while/gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_2/dropout_2/Cast?
 while/gru_cell_2/dropout_2/Mul_1Mul"while/gru_cell_2/dropout_2/Mul:z:0#while/gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_2/dropout_2/Mul_1?
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_2/ReadVariableOp?
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_2/unstack?
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_1?
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack?
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice/stack_1?
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2?
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice?
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul?
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_2?
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_1/stack?
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_1/stack_1?
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2?
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1?
while/gru_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_1?
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_3?
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_2/strided_slice_2/stack?
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1?
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2?
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2?
while/gru_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_2?
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack?
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_3/stack_1?
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2?
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_3?
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd?
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_4/stack?
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_4/stack_1?
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2?
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_4?
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_1?
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_5/stack?
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1?
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2?
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_2/strided_slice_5?
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_2?
while/gru_cell_2/mulMulwhile_placeholder_2"while/gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul?
while/gru_cell_2/mul_1Mulwhile_placeholder_2$while/gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_1?
while/gru_cell_2/mul_2Mulwhile_placeholder_2$while/gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_2?
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_4?
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack?
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_2/strided_slice_6/stack_1?
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2?
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6?
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_3?
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_5?
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_7/stack?
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_7/stack_1?
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2?
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7?
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_4?
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack?
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_8/stack_1?
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2?
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_8?
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_3?
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_9/stack?
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_9/stack_1?
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2?
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_9?
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_4?
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add?
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid?
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_1?
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid_1?
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_6?
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_2/strided_slice_10/stack?
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1?
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2?
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10?
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_2:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_5?
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_2/strided_slice_11/stack?
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1?
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2?
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_2/strided_slice_11?
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_5?
while/gru_cell_2/mul_3Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_3?
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_2?
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Tanh?
while/gru_cell_2/mul_4Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_4u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_2/sub/x?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/sub?
while/gru_cell_2/mul_5Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_5?
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_4:z:0while/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 
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
while_cond_565934
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_565934___redundant_placeholder04
0while_while_cond_565934___redundant_placeholder14
0while_while_cond_565934___redundant_placeholder24
0while_while_cond_565934___redundant_placeholder3
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
?
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_567201

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
seed2???2(
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
seed2???2(
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
?
?
while_cond_564379
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_564379___redundant_placeholder04
0while_while_cond_564379___redundant_placeholder14
0while_while_cond_564379___redundant_placeholder24
0while_while_cond_564379___redundant_placeholder3
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
while_body_566230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_2_readvariableop_resource_00
,while_gru_cell_2_readvariableop_1_resource_00
,while_gru_cell_2_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_2_readvariableop_resource.
*while_gru_cell_2_readvariableop_1_resource.
*while_gru_cell_2_readvariableop_4_resource??while/gru_cell_2/ReadVariableOp?!while/gru_cell_2/ReadVariableOp_1?!while/gru_cell_2/ReadVariableOp_2?!while/gru_cell_2/ReadVariableOp_3?!while/gru_cell_2/ReadVariableOp_4?!while/gru_cell_2/ReadVariableOp_5?!while/gru_cell_2/ReadVariableOp_6?
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
 while/gru_cell_2/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape?
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_2/ones_like/Const?
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/ones_like?
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_2/ReadVariableOp?
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_2/unstack?
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_1?
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack?
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice/stack_1?
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2?
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice?
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul?
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_2?
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_1/stack?
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_1/stack_1?
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2?
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1?
while/gru_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_1?
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_3?
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_2/strided_slice_2/stack?
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1?
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2?
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2?
while/gru_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_2?
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack?
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_3/stack_1?
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2?
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_3?
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd?
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_4/stack?
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_4/stack_1?
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2?
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_4?
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_1?
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_5/stack?
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1?
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2?
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_2/strided_slice_5?
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_2?
while/gru_cell_2/mulMulwhile_placeholder_2#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul?
while/gru_cell_2/mul_1Mulwhile_placeholder_2#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_1?
while/gru_cell_2/mul_2Mulwhile_placeholder_2#while/gru_cell_2/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_2?
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_4?
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack?
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_2/strided_slice_6/stack_1?
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2?
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6?
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_3?
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_5?
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_7/stack?
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_7/stack_1?
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2?
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7?
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_4?
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack?
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_8/stack_1?
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2?
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_8?
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_3?
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_9/stack?
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_9/stack_1?
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2?
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_9?
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_4?
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add?
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid?
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_1?
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid_1?
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_6?
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_2/strided_slice_10/stack?
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1?
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2?
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10?
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_2:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_5?
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_2/strided_slice_11/stack?
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1?
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2?
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_2/strided_slice_11?
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_5?
while/gru_cell_2/mul_3Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_3?
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_2?
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Tanh?
while/gru_cell_2/mul_4Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_4u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_2/sub/x?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/sub?
while/gru_cell_2/mul_5Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_5?
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_4:z:0while/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 
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
?
?
(__inference_model_4_layer_call_fn_564999
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*2
config_proto" 

CPU

GPU2*3,4J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_5649822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
??
?
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_563682

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
seed???)*
seed2?? 2&
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
seed2ᠾ2(
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
seed2??2(
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
?
?
(__inference_model_4_layer_call_fn_565039
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*2
config_proto" 

CPU

GPU2*3,4J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_5650222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_5
??
?
while_body_564380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_2_readvariableop_resource_00
,while_gru_cell_2_readvariableop_1_resource_00
,while_gru_cell_2_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_2_readvariableop_resource.
*while_gru_cell_2_readvariableop_1_resource.
*while_gru_cell_2_readvariableop_4_resource??while/gru_cell_2/ReadVariableOp?!while/gru_cell_2/ReadVariableOp_1?!while/gru_cell_2/ReadVariableOp_2?!while/gru_cell_2/ReadVariableOp_3?!while/gru_cell_2/ReadVariableOp_4?!while/gru_cell_2/ReadVariableOp_5?!while/gru_cell_2/ReadVariableOp_6?
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
 while/gru_cell_2/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell_2/ones_like/Shape?
 while/gru_cell_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell_2/ones_like/Const?
while/gru_cell_2/ones_likeFill)while/gru_cell_2/ones_like/Shape:output:0)while/gru_cell_2/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/ones_like?
while/gru_cell_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell_2/dropout/Const?
while/gru_cell_2/dropout/MulMul#while/gru_cell_2/ones_like:output:0'while/gru_cell_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/dropout/Mul?
while/gru_cell_2/dropout/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell_2/dropout/Shape?
5while/gru_cell_2/dropout/random_uniform/RandomUniformRandomUniform'while/gru_cell_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ס827
5while/gru_cell_2/dropout/random_uniform/RandomUniform?
'while/gru_cell_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell_2/dropout/GreaterEqual/y?
%while/gru_cell_2/dropout/GreaterEqualGreaterEqual>while/gru_cell_2/dropout/random_uniform/RandomUniform:output:00while/gru_cell_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell_2/dropout/GreaterEqual?
while/gru_cell_2/dropout/CastCast)while/gru_cell_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell_2/dropout/Cast?
while/gru_cell_2/dropout/Mul_1Mul while/gru_cell_2/dropout/Mul:z:0!while/gru_cell_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_2/dropout/Mul_1?
 while/gru_cell_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_1/Const?
while/gru_cell_2/dropout_1/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_2/dropout_1/Mul?
 while/gru_cell_2/dropout_1/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_1/Shape?
7while/gru_cell_2/dropout_1/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??,29
7while/gru_cell_2/dropout_1/random_uniform/RandomUniform?
)while/gru_cell_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_2/dropout_1/GreaterEqual/y?
'while/gru_cell_2/dropout_1/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_1/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_2/dropout_1/GreaterEqual?
while/gru_cell_2/dropout_1/CastCast+while/gru_cell_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_2/dropout_1/Cast?
 while/gru_cell_2/dropout_1/Mul_1Mul"while/gru_cell_2/dropout_1/Mul:z:0#while/gru_cell_2/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_2/dropout_1/Mul_1?
 while/gru_cell_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 while/gru_cell_2/dropout_2/Const?
while/gru_cell_2/dropout_2/MulMul#while/gru_cell_2/ones_like:output:0)while/gru_cell_2/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell_2/dropout_2/Mul?
 while/gru_cell_2/dropout_2/ShapeShape#while/gru_cell_2/ones_like:output:0*
T0*
_output_shapes
:2"
 while/gru_cell_2/dropout_2/Shape?
7while/gru_cell_2/dropout_2/random_uniform/RandomUniformRandomUniform)while/gru_cell_2/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???29
7while/gru_cell_2/dropout_2/random_uniform/RandomUniform?
)while/gru_cell_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)while/gru_cell_2/dropout_2/GreaterEqual/y?
'while/gru_cell_2/dropout_2/GreaterEqualGreaterEqual@while/gru_cell_2/dropout_2/random_uniform/RandomUniform:output:02while/gru_cell_2/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/gru_cell_2/dropout_2/GreaterEqual?
while/gru_cell_2/dropout_2/CastCast+while/gru_cell_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
while/gru_cell_2/dropout_2/Cast?
 while/gru_cell_2/dropout_2/Mul_1Mul"while/gru_cell_2/dropout_2/Mul:z:0#while/gru_cell_2/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 while/gru_cell_2/dropout_2/Mul_1?
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes
:	?*
dtype02!
while/gru_cell_2/ReadVariableOp?
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_2/unstack?
!while/gru_cell_2/ReadVariableOp_1ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_1?
$while/gru_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell_2/strided_slice/stack?
&while/gru_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice/stack_1?
&while/gru_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell_2/strided_slice/stack_2?
while/gru_cell_2/strided_sliceStridedSlice)while/gru_cell_2/ReadVariableOp_1:value:0-while/gru_cell_2/strided_slice/stack:output:0/while/gru_cell_2/strided_slice/stack_1:output:0/while/gru_cell_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell_2/strided_slice?
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul?
!while/gru_cell_2/ReadVariableOp_2ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_2?
&while/gru_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_1/stack?
(while/gru_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_1/stack_1?
(while/gru_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_1/stack_2?
 while/gru_cell_2/strided_slice_1StridedSlice)while/gru_cell_2/ReadVariableOp_2:value:0/while/gru_cell_2/strided_slice_1/stack:output:01while/gru_cell_2/strided_slice_1/stack_1:output:01while/gru_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_1?
while/gru_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_1?
!while/gru_cell_2/ReadVariableOp_3ReadVariableOp,while_gru_cell_2_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02#
!while/gru_cell_2/ReadVariableOp_3?
&while/gru_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell_2/strided_slice_2/stack?
(while/gru_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/gru_cell_2/strided_slice_2/stack_1?
(while/gru_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_2/stack_2?
 while/gru_cell_2/strided_slice_2StridedSlice)while/gru_cell_2/ReadVariableOp_3:value:0/while/gru_cell_2/strided_slice_2/stack:output:01while/gru_cell_2/strided_slice_2/stack_1:output:01while/gru_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_2?
while/gru_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/gru_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_2?
&while/gru_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_3/stack?
(while/gru_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_3/stack_1?
(while/gru_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_3/stack_2?
 while/gru_cell_2/strided_slice_3StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_3/stack:output:01while/gru_cell_2/strided_slice_3/stack_1:output:01while/gru_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_3?
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0)while/gru_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd?
&while/gru_cell_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_4/stack?
(while/gru_cell_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_4/stack_1?
(while/gru_cell_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_4/stack_2?
 while/gru_cell_2/strided_slice_4StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_4/stack:output:01while/gru_cell_2/strided_slice_4/stack_1:output:01while/gru_cell_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_4?
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0)while/gru_cell_2/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_1?
&while/gru_cell_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_5/stack?
(while/gru_cell_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(while/gru_cell_2/strided_slice_5/stack_1?
(while/gru_cell_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_5/stack_2?
 while/gru_cell_2/strided_slice_5StridedSlice!while/gru_cell_2/unstack:output:0/while/gru_cell_2/strided_slice_5/stack:output:01while/gru_cell_2/strided_slice_5/stack_1:output:01while/gru_cell_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2"
 while/gru_cell_2/strided_slice_5?
while/gru_cell_2/BiasAdd_2BiasAdd#while/gru_cell_2/MatMul_2:product:0)while/gru_cell_2/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_2?
while/gru_cell_2/mulMulwhile_placeholder_2"while/gru_cell_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul?
while/gru_cell_2/mul_1Mulwhile_placeholder_2$while/gru_cell_2/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_1?
while/gru_cell_2/mul_2Mulwhile_placeholder_2$while/gru_cell_2/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_2?
!while/gru_cell_2/ReadVariableOp_4ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_4?
&while/gru_cell_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell_2/strided_slice_6/stack?
(while/gru_cell_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(while/gru_cell_2/strided_slice_6/stack_1?
(while/gru_cell_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_6/stack_2?
 while/gru_cell_2/strided_slice_6StridedSlice)while/gru_cell_2/ReadVariableOp_4:value:0/while/gru_cell_2/strided_slice_6/stack:output:01while/gru_cell_2/strided_slice_6/stack_1:output:01while/gru_cell_2/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_6?
while/gru_cell_2/MatMul_3MatMulwhile/gru_cell_2/mul:z:0)while/gru_cell_2/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_3?
!while/gru_cell_2/ReadVariableOp_5ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_5?
&while/gru_cell_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell_2/strided_slice_7/stack?
(while/gru_cell_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(while/gru_cell_2/strided_slice_7/stack_1?
(while/gru_cell_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/gru_cell_2/strided_slice_7/stack_2?
 while/gru_cell_2/strided_slice_7StridedSlice)while/gru_cell_2/ReadVariableOp_5:value:0/while/gru_cell_2/strided_slice_7/stack:output:01while/gru_cell_2/strided_slice_7/stack_1:output:01while/gru_cell_2/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 while/gru_cell_2/strided_slice_7?
while/gru_cell_2/MatMul_4MatMulwhile/gru_cell_2/mul_1:z:0)while/gru_cell_2/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_4?
&while/gru_cell_2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell_2/strided_slice_8/stack?
(while/gru_cell_2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_8/stack_1?
(while/gru_cell_2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_8/stack_2?
 while/gru_cell_2/strided_slice_8StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_8/stack:output:01while/gru_cell_2/strided_slice_8/stack_1:output:01while/gru_cell_2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2"
 while/gru_cell_2/strided_slice_8?
while/gru_cell_2/BiasAdd_3BiasAdd#while/gru_cell_2/MatMul_3:product:0)while/gru_cell_2/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_3?
&while/gru_cell_2/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell_2/strided_slice_9/stack?
(while/gru_cell_2/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2*
(while/gru_cell_2/strided_slice_9/stack_1?
(while/gru_cell_2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(while/gru_cell_2/strided_slice_9/stack_2?
 while/gru_cell_2/strided_slice_9StridedSlice!while/gru_cell_2/unstack:output:1/while/gru_cell_2/strided_slice_9/stack:output:01while/gru_cell_2/strided_slice_9/stack_1:output:01while/gru_cell_2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2"
 while/gru_cell_2/strided_slice_9?
while/gru_cell_2/BiasAdd_4BiasAdd#while/gru_cell_2/MatMul_4:product:0)while/gru_cell_2/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_4?
while/gru_cell_2/addAddV2!while/gru_cell_2/BiasAdd:output:0#while/gru_cell_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add?
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid?
while/gru_cell_2/add_1AddV2#while/gru_cell_2/BiasAdd_1:output:0#while/gru_cell_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_1?
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Sigmoid_1?
!while/gru_cell_2/ReadVariableOp_6ReadVariableOp,while_gru_cell_2_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02#
!while/gru_cell_2/ReadVariableOp_6?
'while/gru_cell_2/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/gru_cell_2/strided_slice_10/stack?
)while/gru_cell_2/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/gru_cell_2/strided_slice_10/stack_1?
)while/gru_cell_2/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/gru_cell_2/strided_slice_10/stack_2?
!while/gru_cell_2/strided_slice_10StridedSlice)while/gru_cell_2/ReadVariableOp_6:value:00while/gru_cell_2/strided_slice_10/stack:output:02while/gru_cell_2/strided_slice_10/stack_1:output:02while/gru_cell_2/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/gru_cell_2/strided_slice_10?
while/gru_cell_2/MatMul_5MatMulwhile/gru_cell_2/mul_2:z:0*while/gru_cell_2/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/MatMul_5?
'while/gru_cell_2/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2)
'while/gru_cell_2/strided_slice_11/stack?
)while/gru_cell_2/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)while/gru_cell_2/strided_slice_11/stack_1?
)while/gru_cell_2/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)while/gru_cell_2/strided_slice_11/stack_2?
!while/gru_cell_2/strided_slice_11StridedSlice!while/gru_cell_2/unstack:output:10while/gru_cell_2/strided_slice_11/stack:output:02while/gru_cell_2/strided_slice_11/stack_1:output:02while/gru_cell_2/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2#
!while/gru_cell_2/strided_slice_11?
while/gru_cell_2/BiasAdd_5BiasAdd#while/gru_cell_2/MatMul_5:product:0*while/gru_cell_2/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/BiasAdd_5?
while/gru_cell_2/mul_3Mulwhile/gru_cell_2/Sigmoid_1:y:0#while/gru_cell_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_3?
while/gru_cell_2/add_2AddV2#while/gru_cell_2/BiasAdd_2:output:0while/gru_cell_2/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_2?
while/gru_cell_2/TanhTanhwhile/gru_cell_2/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/Tanh?
while/gru_cell_2/mul_4Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_4u
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_2/sub/x?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/sub?
while/gru_cell_2/mul_5Mulwhile/gru_cell_2/sub:z:0while/gru_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/mul_5?
while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_4:z:0while/gru_cell_2/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_2/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0 ^while/gru_cell_2/ReadVariableOp"^while/gru_cell_2/ReadVariableOp_1"^while/gru_cell_2/ReadVariableOp_2"^while/gru_cell_2/ReadVariableOp_3"^while/gru_cell_2/ReadVariableOp_4"^while/gru_cell_2/ReadVariableOp_5"^while/gru_cell_2/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"Z
*while_gru_cell_2_readvariableop_1_resource,while_gru_cell_2_readvariableop_1_resource_0"Z
*while_gru_cell_2_readvariableop_4_resource,while_gru_cell_2_readvariableop_4_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp2F
!while/gru_cell_2/ReadVariableOp_1!while/gru_cell_2/ReadVariableOp_12F
!while/gru_cell_2/ReadVariableOp_2!while/gru_cell_2/ReadVariableOp_22F
!while/gru_cell_2/ReadVariableOp_3!while/gru_cell_2/ReadVariableOp_32F
!while/gru_cell_2/ReadVariableOp_4!while/gru_cell_2/ReadVariableOp_42F
!while/gru_cell_2/ReadVariableOp_5!while/gru_cell_2/ReadVariableOp_52F
!while/gru_cell_2/ReadVariableOp_6!while/gru_cell_2/ReadVariableOp_6: 
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
D__inference_dense_12_layer_call_and_return_conditional_losses_564920

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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_54
serving_default_input_5:0?????????	<
dense_120
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?M
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
*_&call_and_return_all_conditional_losses
`_default_save_signature
a__call__"?K
_tf_keras_network?J{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}, "name": "gru_2", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_2", "inbound_nodes": [[["gru_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["layer_normalization_2", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["dense_12", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}, "name": "gru_2", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_2", "inbound_nodes": [[["gru_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["layer_normalization_2", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["dense_12", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 5.000000328436727e-06, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?

_tf_keras_rnn_layer?	{"class_name": "GRU", "name": "gru_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 9]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 9]}}
?
axis
	gamma
beta
regularization_losses
	variables
trainable_variables
	keras_api
*d&call_and_return_all_conditional_losses
e__call__"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*f&call_and_return_all_conditional_losses
g__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
iter

beta_1

 beta_2
	!decay
"learning_rate
#momentum_cachemQmRmSmT$mU%mV&mWvXvYvZv[$v\%v]&v^"
	optimizer
 "
trackable_list_wrapper
Q
$0
%1
&2
3
4
5
6"
trackable_list_wrapper
Q
$0
%1
&2
3
4
5
6"
trackable_list_wrapper
?
regularization_losses
'layer_regularization_losses
	variables
(metrics
trainable_variables
)layer_metrics
*non_trainable_variables

+layers
a__call__
`_default_save_signature
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
,
hserving_default"
signature_map
?

$kernel
%recurrent_kernel
&bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
*i&call_and_return_all_conditional_losses
j__call__"?
_tf_keras_layer?{"class_name": "GRUCell", "name": "gru_cell_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_2", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
?
regularization_losses
0layer_regularization_losses
	variables
1metrics
trainable_variables

2states
3layer_metrics
4non_trainable_variables

5layers
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(?2layer_normalization_2/gamma
):'?2layer_normalization_2/beta
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
6layer_regularization_losses
regularization_losses
	variables
7metrics
trainable_variables
8layer_metrics
9non_trainable_variables

:layers
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_12/kernel
:2dense_12/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
;layer_regularization_losses
regularization_losses
	variables
<metrics
trainable_variables
=layer_metrics
>non_trainable_variables

?layers
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
*:(		?2gru_2/gru_cell_2/kernel
5:3
??2!gru_2/gru_cell_2/recurrent_kernel
(:&	?2gru_2/gru_cell_2/bias
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
?
Blayer_regularization_losses
,regularization_losses
-	variables
Cmetrics
.trainable_variables
Dlayer_metrics
Enon_trainable_variables

Flayers
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
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
'
0"
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
?
	Gtotal
	Hcount
I	variables
J	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?"
Ktrue_positives
Ltrue_negatives
Mfalse_positives
Nfalse_negatives
O	variables
P	keras_api"?!
_tf_keras_metric?!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
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
:  (2total
:  (2count
.
G0
H1"
trackable_list_wrapper
-
I	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
<
K0
L1
M2
N3"
trackable_list_wrapper
-
O	variables"
_generic_user_object
0:.?2#Nadam/layer_normalization_2/gamma/m
/:-?2"Nadam/layer_normalization_2/beta/m
(:&	?2Nadam/dense_12/kernel/m
!:2Nadam/dense_12/bias/m
0:.		?2Nadam/gru_2/gru_cell_2/kernel/m
;:9
??2)Nadam/gru_2/gru_cell_2/recurrent_kernel/m
.:,	?2Nadam/gru_2/gru_cell_2/bias/m
0:.?2#Nadam/layer_normalization_2/gamma/v
/:-?2"Nadam/layer_normalization_2/beta/v
(:&	?2Nadam/dense_12/kernel/v
!:2Nadam/dense_12/bias/v
0:.		?2Nadam/gru_2/gru_cell_2/kernel/v
;:9
??2)Nadam/gru_2/gru_cell_2/recurrent_kernel/v
.:,	?2Nadam/gru_2/gru_cell_2/bias/v
?2?
C__inference_model_4_layer_call_and_return_conditional_losses_564958
C__inference_model_4_layer_call_and_return_conditional_losses_565748
C__inference_model_4_layer_call_and_return_conditional_losses_565432
C__inference_model_4_layer_call_and_return_conditional_losses_564937?
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
?2?
!__inference__wrapped_model_563530?
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
annotations? **?'
%?"
input_5?????????	
?2?
(__inference_model_4_layer_call_fn_565767
(__inference_model_4_layer_call_fn_565039
(__inference_model_4_layer_call_fn_565786
(__inference_model_4_layer_call_fn_564999?
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
?2?
A__inference_gru_2_layer_call_and_return_conditional_losses_566105
A__inference_gru_2_layer_call_and_return_conditional_losses_566717
A__inference_gru_2_layer_call_and_return_conditional_losses_566988
A__inference_gru_2_layer_call_and_return_conditional_losses_566376?
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
?2?
&__inference_gru_2_layer_call_fn_566999
&__inference_gru_2_layer_call_fn_567010
&__inference_gru_2_layer_call_fn_566387
&__inference_gru_2_layer_call_fn_566398?
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
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_567052?
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
6__inference_layer_normalization_2_layer_call_fn_567061?
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
D__inference_dense_12_layer_call_and_return_conditional_losses_567072?
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
)__inference_dense_12_layer_call_fn_567081?
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
$__inference_signature_wrapper_565068input_5"?
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
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_567297
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_567201?
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
+__inference_gru_cell_2_layer_call_fn_567311
+__inference_gru_cell_2_layer_call_fn_567325?
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
!__inference__wrapped_model_563530t&$%4?1
*?'
%?"
input_5?????????	
? "3?0
.
dense_12"?
dense_12??????????
D__inference_dense_12_layer_call_and_return_conditional_losses_567072]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_12_layer_call_fn_567081P0?-
&?#
!?
inputs??????????
? "???????????
A__inference_gru_2_layer_call_and_return_conditional_losses_566105~&$%O?L
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
A__inference_gru_2_layer_call_and_return_conditional_losses_566376~&$%O?L
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
A__inference_gru_2_layer_call_and_return_conditional_losses_566717n&$%??<
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
A__inference_gru_2_layer_call_and_return_conditional_losses_566988n&$%??<
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
&__inference_gru_2_layer_call_fn_566387q&$%O?L
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
&__inference_gru_2_layer_call_fn_566398q&$%O?L
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
&__inference_gru_2_layer_call_fn_566999a&$%??<
5?2
$?!
inputs?????????	

 
p

 
? "????????????
&__inference_gru_2_layer_call_fn_567010a&$%??<
5?2
$?!
inputs?????????	

 
p 

 
? "????????????
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_567201?&$%]?Z
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
F__inference_gru_cell_2_layer_call_and_return_conditional_losses_567297?&$%]?Z
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
+__inference_gru_cell_2_layer_call_fn_567311?&$%]?Z
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
+__inference_gru_cell_2_layer_call_fn_567325?&$%]?Z
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
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_567052^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
6__inference_layer_normalization_2_layer_call_fn_567061Q0?-
&?#
!?
inputs??????????
? "????????????
C__inference_model_4_layer_call_and_return_conditional_losses_564937n&$%<?9
2?/
%?"
input_5?????????	
p

 
? "%?"
?
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_564958n&$%<?9
2?/
%?"
input_5?????????	
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_565432m&$%;?8
1?.
$?!
inputs?????????	
p

 
? "%?"
?
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_565748m&$%;?8
1?.
$?!
inputs?????????	
p 

 
? "%?"
?
0?????????
? ?
(__inference_model_4_layer_call_fn_564999a&$%<?9
2?/
%?"
input_5?????????	
p

 
? "???????????
(__inference_model_4_layer_call_fn_565039a&$%<?9
2?/
%?"
input_5?????????	
p 

 
? "???????????
(__inference_model_4_layer_call_fn_565767`&$%;?8
1?.
$?!
inputs?????????	
p

 
? "???????????
(__inference_model_4_layer_call_fn_565786`&$%;?8
1?.
$?!
inputs?????????	
p 

 
? "???????????
$__inference_signature_wrapper_565068&$%??<
? 
5?2
0
input_5%?"
input_5?????????	"3?0
.
dense_12"?
dense_12?????????