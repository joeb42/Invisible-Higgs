??)
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
?"serve*2.4.12unknown8??'
?
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelayer_normalization/gamma
?
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes	
:?*
dtype0
?
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelayer_normalization/beta
?
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes	
:?*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
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
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:		?*
dtype0
?
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*.
shared_namegru/gru_cell/recurrent_kernel
?
1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel* 
_output_shapes
:
??*
dtype0

gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*"
shared_namegru/gru_cell/bias
x
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
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
!Nadam/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Nadam/layer_normalization/gamma/m
?
5Nadam/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp!Nadam/layer_normalization/gamma/m*
_output_shapes	
:?*
dtype0
?
 Nadam/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Nadam/layer_normalization/beta/m
?
4Nadam/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp Nadam/layer_normalization/beta/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameNadam/dense/kernel/m
~
(Nadam/dense/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/m*
_output_shapes
:	?*
dtype0
|
Nadam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameNadam/dense/bias/m
u
&Nadam/dense/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense/bias/m*
_output_shapes
:*
dtype0
?
Nadam/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*,
shared_nameNadam/gru/gru_cell/kernel/m
?
/Nadam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOpNadam/gru/gru_cell/kernel/m*
_output_shapes
:		?*
dtype0
?
%Nadam/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%Nadam/gru/gru_cell/recurrent_kernel/m
?
9Nadam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp%Nadam/gru/gru_cell/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameNadam/gru/gru_cell/bias/m
?
-Nadam/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOpNadam/gru/gru_cell/bias/m*
_output_shapes
:	?*
dtype0
?
!Nadam/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Nadam/layer_normalization/gamma/v
?
5Nadam/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp!Nadam/layer_normalization/gamma/v*
_output_shapes	
:?*
dtype0
?
 Nadam/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Nadam/layer_normalization/beta/v
?
4Nadam/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp Nadam/layer_normalization/beta/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameNadam/dense/kernel/v
~
(Nadam/dense/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/v*
_output_shapes
:	?*
dtype0
|
Nadam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameNadam/dense/bias/v
u
&Nadam/dense/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense/bias/v*
_output_shapes
:*
dtype0
?
Nadam/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		?*,
shared_nameNadam/gru/gru_cell/kernel/v
?
/Nadam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOpNadam/gru/gru_cell/kernel/v*
_output_shapes
:		?*
dtype0
?
%Nadam/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%Nadam/gru/gru_cell/recurrent_kernel/v
?
9Nadam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp%Nadam/gru/gru_cell/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameNadam/gru/gru_cell/bias/v
?
-Nadam/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOpNadam/gru/gru_cell/bias/v*
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
trainable_variables
	variables
regularization_losses
		keras_api


signatures
 
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
q
axis
	gamma
beta
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?
iter

beta_1

 beta_2
	!decay
"learning_rate
#momentum_cachemQmRmSmT$mU%mV&mWvXvYvZv[$v\%v]&v^
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
 
?
'metrics

(layers
)layer_metrics
trainable_variables
*non_trainable_variables
	variables
+layer_regularization_losses
regularization_losses
 
~

$kernel
%recurrent_kernel
&bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
 

$0
%1
&2

$0
%1
&2
 
?

0states
1metrics

2layers
3layer_metrics
trainable_variables
4non_trainable_variables
	variables
5layer_regularization_losses
regularization_losses
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
6metrics

7layers
8layer_metrics
trainable_variables
9non_trainable_variables
	variables
:layer_regularization_losses
regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
;metrics

<layers
=layer_metrics
trainable_variables
>non_trainable_variables
	variables
?layer_regularization_losses
regularization_losses
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
YW
VARIABLE_VALUEgru/gru_cell/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEgru/gru_cell/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEgru/gru_cell/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

0
1
2
3
 
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
 
?
Bmetrics

Clayers
Dlayer_metrics
,trainable_variables
Enon_trainable_variables
-	variables
Flayer_regularization_losses
.regularization_losses
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
VARIABLE_VALUE!Nadam/layer_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Nadam/layer_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUENadam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/gru/gru_cell/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Nadam/gru/gru_cell/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/gru/gru_cell/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Nadam/layer_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Nadam/layer_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUENadam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/gru/gru_cell/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Nadam/gru/gru_cell/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/gru/gru_cell/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????	*
dtype0* 
shape:?????????	
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1gru/gru_cell/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernellayer_normalization/gammalayer_normalization/betadense/kernel
dense/bias*
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
GPU2*4,5J 8? *-
f(R&
$__inference_signature_wrapper_548652
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp5Nadam/layer_normalization/gamma/m/Read/ReadVariableOp4Nadam/layer_normalization/beta/m/Read/ReadVariableOp(Nadam/dense/kernel/m/Read/ReadVariableOp&Nadam/dense/bias/m/Read/ReadVariableOp/Nadam/gru/gru_cell/kernel/m/Read/ReadVariableOp9Nadam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOp-Nadam/gru/gru_cell/bias/m/Read/ReadVariableOp5Nadam/layer_normalization/gamma/v/Read/ReadVariableOp4Nadam/layer_normalization/beta/v/Read/ReadVariableOp(Nadam/dense/kernel/v/Read/ReadVariableOp&Nadam/dense/bias/v/Read/ReadVariableOp/Nadam/gru/gru_cell/kernel/v/Read/ReadVariableOp9Nadam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOp-Nadam/gru/gru_cell/bias/v/Read/ReadVariableOpConst*.
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
GPU2*4,5J 8? *(
f#R!
__inference__traced_save_551031
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betadense/kernel
dense/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachegru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negatives!Nadam/layer_normalization/gamma/m Nadam/layer_normalization/beta/mNadam/dense/kernel/mNadam/dense/bias/mNadam/gru/gru_cell/kernel/m%Nadam/gru/gru_cell/recurrent_kernel/mNadam/gru/gru_cell/bias/m!Nadam/layer_normalization/gamma/v Nadam/layer_normalization/beta/vNadam/dense/kernel/vNadam/dense/bias/vNadam/gru/gru_cell/kernel/v%Nadam/gru/gru_cell/recurrent_kernel/vNadam/gru/gru_cell/bias/v*-
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
GPU2*4,5J 8? *+
f&R$
"__inference__traced_restore_551140??&
?
?
&__inference_model_layer_call_fn_548623
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU2*4,5J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5486062
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
_user_specified_name	input_1
??
?
?__inference_gru_layer_call_and_return_conditional_losses_548405

inputs$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileD
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
strided_slice_2r
gru_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_like?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_2?
gru_cell/mulMulzeros:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulzeros:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulzeros:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul:z:0!gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0"gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_5?
gru_cell/mul_3Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_3?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Tanh?
gru_cell/mul_4Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_4e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub
gru_cell/mul_5Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_5?
gru_cell/add_3AddV2gru_cell/mul_4:z:0gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_548259*
condR
while_cond_548258*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
?__inference_gru_layer_call_and_return_conditional_losses_550301

inputs$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileD
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
strided_slice_2r
gru_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout/Const?
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shape?
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ˈ2/
-gru_cell/dropout/random_uniform/RandomUniform?
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
gru_cell/dropout/GreaterEqual/y?
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/GreaterEqual?
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout/Cast?
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_1/Const?
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shape?
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ʮ?21
/gru_cell/dropout_1/random_uniform/RandomUniform?
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell/dropout_1/GreaterEqual/y?
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_1/GreaterEqual?
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_1/Cast?
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_2/Const?
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shape?
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_2/random_uniform/RandomUniform?
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell/dropout_2/GreaterEqual/y?
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_2/GreaterEqual?
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_2/Cast?
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_2?
gru_cell/mulMulzeros:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulzeros:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulzeros:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul:z:0!gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0"gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_5?
gru_cell/mul_3Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_3?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Tanh?
gru_cell/mul_4Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_4e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub
gru_cell/mul_5Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_5?
gru_cell/add_3AddV2gru_cell/mul_4:z:0gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_550131*
condR
while_cond_550130*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
D__inference_gru_cell_layer_call_and_return_conditional_losses_547266

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
seed2Ȝ?2(
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
?h
?
D__inference_gru_cell_layer_call_and_return_conditional_losses_550881

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
?
?
while_cond_547963
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_547963___redundant_placeholder04
0while_while_cond_547963___redundant_placeholder14
0while_while_cond_547963___redundant_placeholder24
0while_while_cond_547963___redundant_placeholder3
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
?
?
$__inference_signature_wrapper_548652
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU2*4,5J 8? **
f%R#
!__inference__wrapped_model_5471142
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
_user_specified_name	input_1
?
?
while_cond_549813
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_549813___redundant_placeholder04
0while_while_cond_549813___redundant_placeholder14
0while_while_cond_549813___redundant_placeholder24
0while_while_cond_549813___redundant_placeholder3
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
?!
?
while_body_547621
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_547643_0
while_gru_cell_547645_0
while_gru_cell_547647_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_547643
while_gru_cell_547645
while_gru_cell_547647??&while/gru_cell/StatefulPartitionedCall?
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
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_547643_0while_gru_cell_547645_0while_gru_cell_547647_0*
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
GPU2*4,5J 8? *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_5472662(
&while/gru_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4"0
while_gru_cell_547643while_gru_cell_547643_0"0
while_gru_cell_547645while_gru_cell_547645_0"0
while_gru_cell_547647while_gru_cell_547647_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 
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
?
?__inference_gru_layer_call_and_return_conditional_losses_548134

inputs$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileD
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
strided_slice_2r
gru_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout/Const?
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shape?
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2/
-gru_cell/dropout/random_uniform/RandomUniform?
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
gru_cell/dropout/GreaterEqual/y?
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/GreaterEqual?
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout/Cast?
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_1/Const?
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shape?
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_1/random_uniform/RandomUniform?
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell/dropout_1/GreaterEqual/y?
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_1/GreaterEqual?
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_1/Cast?
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_2/Const?
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shape?
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??P21
/gru_cell/dropout_2/random_uniform/RandomUniform?
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell/dropout_2/GreaterEqual/y?
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_2/GreaterEqual?
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_2/Cast?
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_2?
gru_cell/mulMulzeros:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulzeros:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulzeros:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul:z:0!gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0"gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_5?
gru_cell/mul_3Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_3?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Tanh?
gru_cell/mul_4Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_4e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub
gru_cell/mul_5Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_5?
gru_cell/add_3AddV2gru_cell/mul_4:z:0gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_547964*
condR
while_cond_547963*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
while_cond_548258
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_548258___redundant_placeholder04
0while_while_cond_548258___redundant_placeholder14
0while_while_cond_548258___redundant_placeholder24
0while_while_cond_548258___redundant_placeholder3
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
?
while_body_548259
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
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
while/gru_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/mulMulwhile_placeholder_2!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mulwhile_placeholder_2!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mulwhile_placeholder_2!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0(while/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_3Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_3?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Tanh?
while/gru_cell/mul_4Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_4q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_5Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_5?
while/gru_cell/add_3AddV2while/gru_cell/mul_4:z:0while/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
&__inference_model_layer_call_fn_549370

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
GPU2*4,5J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5486062
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
??
?
while_body_550426
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
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
while/gru_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/mulMulwhile_placeholder_2!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mulwhile_placeholder_2!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mulwhile_placeholder_2!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0(while/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_3Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_3?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Tanh?
while/gru_cell/mul_4Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_4q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_5Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_5?
while/gru_cell/add_3AddV2while/gru_cell/mul_4:z:0while/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
while_cond_550425
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_550425___redundant_placeholder04
0while_while_cond_550425___redundant_placeholder14
0while_while_cond_550425___redundant_placeholder24
0while_while_cond_550425___redundant_placeholder3
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
?
?
while_cond_549518
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_549518___redundant_placeholder04
0while_while_cond_549518___redundant_placeholder14
0while_while_cond_549518___redundant_placeholder24
0while_while_cond_549518___redundant_placeholder3
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
?
?
&__inference_model_layer_call_fn_548583
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU2*4,5J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5485662
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
_user_specified_name	input_1
??
?
D__inference_gru_cell_layer_call_and_return_conditional_losses_550785

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
seed???)*
seed2??m2&
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
seed???)*
seed2??2(
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
seed2???2(
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
A__inference_dense_layer_call_and_return_conditional_losses_548504

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
??
?
gru_while_body_549141$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_00
,gru_while_gru_cell_readvariableop_resource_02
.gru_while_gru_cell_readvariableop_1_resource_02
.gru_while_gru_cell_readvariableop_4_resource_0
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor.
*gru_while_gru_cell_readvariableop_resource0
,gru_while_gru_cell_readvariableop_1_resource0
,gru_while_gru_cell_readvariableop_4_resource??!gru/while/gru_cell/ReadVariableOp?#gru/while/gru_cell/ReadVariableOp_1?#gru/while/gru_cell/ReadVariableOp_2?#gru/while/gru_cell/ReadVariableOp_3?#gru/while/gru_cell/ReadVariableOp_4?#gru/while/gru_cell/ReadVariableOp_5?#gru/while/gru_cell/ReadVariableOp_6?
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem?
"gru/while/gru_cell/ones_like/ShapeShapegru_while_placeholder_2*
T0*
_output_shapes
:2$
"gru/while/gru_cell/ones_like/Shape?
"gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru/while/gru_cell/ones_like/Const?
gru/while/gru_cell/ones_likeFill+gru/while/gru_cell/ones_like/Shape:output:0+gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/ones_like?
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02#
!gru/while/gru_cell/ReadVariableOp?
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/while/gru_cell/unstack?
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1?
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stack?
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(gru/while/gru_cell/strided_slice/stack_1?
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2?
 gru/while/gru_cell/strided_sliceStridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_slice?
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0)gru/while/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul?
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2?
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(gru/while/gru_cell/strided_slice_1/stack?
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2,
*gru/while/gru_cell/strided_slice_1/stack_1?
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_1/stack_2?
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1?
gru/while/gru_cell/MatMul_1MatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_1?
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3?
(gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(gru/while/gru_cell/strided_slice_2/stack?
*gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_2/stack_1?
*gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_2/stack_2?
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2?
gru/while/gru_cell/MatMul_2MatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_2?
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stack?
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*gru/while/gru_cell/strided_slice_3/stack_1?
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2?
"gru/while/gru_cell/strided_slice_3StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2$
"gru/while/gru_cell/strided_slice_3?
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd?
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2*
(gru/while/gru_cell/strided_slice_4/stack?
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*gru/while/gru_cell/strided_slice_4/stack_1?
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2?
"gru/while/gru_cell/strided_slice_4StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2$
"gru/while/gru_cell/strided_slice_4?
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_1?
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2*
(gru/while/gru_cell/strided_slice_5/stack?
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1?
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2?
"gru/while/gru_cell/strided_slice_5StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2$
"gru/while/gru_cell/strided_slice_5?
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_2?
gru/while/gru_cell/mulMulgru_while_placeholder_2%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul?
gru/while/gru_cell/mul_1Mulgru_while_placeholder_2%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_1?
gru/while/gru_cell/mul_2Mulgru_while_placeholder_2%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_2?
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4?
(gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_6/stack?
*gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2,
*gru/while/gru_cell/strided_slice_6/stack_1?
*gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_6/stack_2?
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6?
gru/while/gru_cell/MatMul_3MatMulgru/while/gru_cell/mul:z:0+gru/while/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_3?
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5?
(gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(gru/while/gru_cell/strided_slice_7/stack?
*gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2,
*gru/while/gru_cell/strided_slice_7/stack_1?
*gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_7/stack_2?
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7?
gru/while/gru_cell/MatMul_4MatMulgru/while/gru_cell/mul_1:z:0+gru/while/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_4?
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_8/stack?
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*gru/while/gru_cell/strided_slice_8/stack_1?
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_8/stack_2?
"gru/while/gru_cell/strided_slice_8StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2$
"gru/while/gru_cell/strided_slice_8?
gru/while/gru_cell/BiasAdd_3BiasAdd%gru/while/gru_cell/MatMul_3:product:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_3?
(gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2*
(gru/while/gru_cell/strided_slice_9/stack?
*gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*gru/while/gru_cell/strided_slice_9/stack_1?
*gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_9/stack_2?
"gru/while/gru_cell/strided_slice_9StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_9/stack:output:03gru/while/gru_cell/strided_slice_9/stack_1:output:03gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2$
"gru/while/gru_cell/strided_slice_9?
gru/while/gru_cell/BiasAdd_4BiasAdd%gru/while/gru_cell/MatMul_4:product:0+gru/while/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_4?
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add?
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid?
gru/while/gru_cell/add_1AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_1?
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid_1?
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6?
)gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2+
)gru/while/gru_cell/strided_slice_10/stack?
+gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+gru/while/gru_cell/strided_slice_10/stack_1?
+gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+gru/while/gru_cell/strided_slice_10/stack_2?
#gru/while/gru_cell/strided_slice_10StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:02gru/while/gru_cell/strided_slice_10/stack:output:04gru/while/gru_cell/strided_slice_10/stack_1:output:04gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2%
#gru/while/gru_cell/strided_slice_10?
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_2:z:0,gru/while/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_5?
)gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2+
)gru/while/gru_cell/strided_slice_11/stack?
+gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gru/while/gru_cell/strided_slice_11/stack_1?
+gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gru/while/gru_cell/strided_slice_11/stack_2?
#gru/while/gru_cell/strided_slice_11StridedSlice#gru/while/gru_cell/unstack:output:12gru/while/gru_cell/strided_slice_11/stack:output:04gru/while/gru_cell/strided_slice_11/stack_1:output:04gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2%
#gru/while/gru_cell/strided_slice_11?
gru/while/gru_cell/BiasAdd_5BiasAdd%gru/while/gru_cell/MatMul_5:product:0,gru/while/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_5?
gru/while/gru_cell/mul_3Mul gru/while/gru_cell/Sigmoid_1:y:0%gru/while/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_3?
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_2:output:0gru/while/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_2?
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Tanh?
gru/while/gru_cell/mul_4Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_4y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/while/gru_cell/sub/x?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/sub?
gru/while/gru_cell/mul_5Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_5?
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_4:z:0gru/while/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_3?
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y?
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1?
gru/while/IdentityIdentitygru/while/add_1:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity?
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1?
gru/while/Identity_2Identitygru/while/add:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2?
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3?
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
gru/while/Identity_4"^
,gru_while_gru_cell_readvariableop_1_resource.gru_while_gru_cell_readvariableop_1_resource_0"^
,gru_while_gru_cell_readvariableop_4_resource.gru_while_gru_cell_readvariableop_4_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"?
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp2J
#gru/while/gru_cell/ReadVariableOp_1#gru/while/gru_cell/ReadVariableOp_12J
#gru/while/gru_cell/ReadVariableOp_2#gru/while/gru_cell/ReadVariableOp_22J
#gru/while/gru_cell/ReadVariableOp_3#gru/while/gru_cell/ReadVariableOp_32J
#gru/while/gru_cell/ReadVariableOp_4#gru/while/gru_cell/ReadVariableOp_42J
#gru/while/gru_cell/ReadVariableOp_5#gru/while/gru_cell/ReadVariableOp_52J
#gru/while/gru_cell/ReadVariableOp_6#gru/while/gru_cell/ReadVariableOp_6: 
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
O__inference_layer_normalization_layer_call_and_return_conditional_losses_548477

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
?
while_body_549814
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
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
while/gru_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/mulMulwhile_placeholder_2!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mulwhile_placeholder_2!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mulwhile_placeholder_2!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0(while/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_3Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_3?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Tanh?
while/gru_cell/mul_4Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_4q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_5Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_5?
while/gru_cell/add_3AddV2while/gru_cell/mul_4:z:0while/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
?I
?
__inference__traced_save_551031
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop@
<savev2_nadam_layer_normalization_gamma_m_read_readvariableop?
;savev2_nadam_layer_normalization_beta_m_read_readvariableop3
/savev2_nadam_dense_kernel_m_read_readvariableop1
-savev2_nadam_dense_bias_m_read_readvariableop:
6savev2_nadam_gru_gru_cell_kernel_m_read_readvariableopD
@savev2_nadam_gru_gru_cell_recurrent_kernel_m_read_readvariableop8
4savev2_nadam_gru_gru_cell_bias_m_read_readvariableop@
<savev2_nadam_layer_normalization_gamma_v_read_readvariableop?
;savev2_nadam_layer_normalization_beta_v_read_readvariableop3
/savev2_nadam_dense_kernel_v_read_readvariableop1
-savev2_nadam_dense_bias_v_read_readvariableop:
6savev2_nadam_gru_gru_cell_kernel_v_read_readvariableopD
@savev2_nadam_gru_gru_cell_recurrent_kernel_v_read_readvariableop8
4savev2_nadam_gru_gru_cell_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop<savev2_nadam_layer_normalization_gamma_m_read_readvariableop;savev2_nadam_layer_normalization_beta_m_read_readvariableop/savev2_nadam_dense_kernel_m_read_readvariableop-savev2_nadam_dense_bias_m_read_readvariableop6savev2_nadam_gru_gru_cell_kernel_m_read_readvariableop@savev2_nadam_gru_gru_cell_recurrent_kernel_m_read_readvariableop4savev2_nadam_gru_gru_cell_bias_m_read_readvariableop<savev2_nadam_layer_normalization_gamma_v_read_readvariableop;savev2_nadam_layer_normalization_beta_v_read_readvariableop/savev2_nadam_dense_kernel_v_read_readvariableop-savev2_nadam_dense_bias_v_read_readvariableop6savev2_nadam_gru_gru_cell_kernel_v_read_readvariableop@savev2_nadam_gru_gru_cell_recurrent_kernel_v_read_readvariableop4savev2_nadam_gru_gru_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
??
?
A__inference_model_layer_call_and_return_conditional_losses_549332

inputs(
$gru_gru_cell_readvariableop_resource*
&gru_gru_cell_readvariableop_1_resource*
&gru_gru_cell_readvariableop_4_resource5
1layer_normalization_mul_2_readvariableop_resource3
/layer_normalization_add_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?gru/gru_cell/ReadVariableOp?gru/gru_cell/ReadVariableOp_1?gru/gru_cell/ReadVariableOp_2?gru/gru_cell/ReadVariableOp_3?gru/gru_cell/ReadVariableOp_4?gru/gru_cell/ReadVariableOp_5?gru/gru_cell/ReadVariableOp_6?	gru/while?&layer_normalization/add/ReadVariableOp?(layer_normalization/mul_2/ReadVariableOpL
	gru/ShapeShapeinputs*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slicee
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessk
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/packed/1?
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const?
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transposeinputsgru/transpose/perm:output:0*
T0*+
_output_shapes
:?????????	2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
gru/strided_slice_2~
gru/gru_cell/ones_like/ShapeShapegru/zeros:output:0*
T0*
_output_shapes
:2
gru/gru_cell/ones_like/Shape?
gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/ones_like/Const?
gru/gru_cell/ones_likeFill%gru/gru_cell/ones_like/Shape:output:0%gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/ones_like?
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru/gru_cell/ReadVariableOp?
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/gru_cell/unstack?
gru/gru_cell/ReadVariableOp_1ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru/gru_cell/ReadVariableOp_1?
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru/gru_cell/strided_slice/stack?
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru/gru_cell/strided_slice/stack_1?
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru/gru_cell/strided_slice/stack_2?
gru/gru_cell/strided_sliceStridedSlice%gru/gru_cell/ReadVariableOp_1:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice?
gru/gru_cell/MatMulMatMulgru/strided_slice_2:output:0#gru/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul?
gru/gru_cell/ReadVariableOp_2ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru/gru_cell/ReadVariableOp_2?
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru/gru_cell/strided_slice_1/stack?
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2&
$gru/gru_cell/strided_slice_1/stack_1?
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_1/stack_2?
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1?
gru/gru_cell/MatMul_1MatMulgru/strided_slice_2:output:0%gru/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_1?
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru/gru_cell/ReadVariableOp_3?
"gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru/gru_cell/strided_slice_2/stack?
$gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_2/stack_1?
$gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_2/stack_2?
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2?
gru/gru_cell/MatMul_2MatMulgru/strided_slice_2:output:0%gru/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_2?
"gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_3/stack?
$gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2&
$gru/gru_cell/strided_slice_3/stack_1?
$gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_2?
gru/gru_cell/strided_slice_3StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru/gru_cell/strided_slice_3?
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd?
"gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2$
"gru/gru_cell/strided_slice_4/stack?
$gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2&
$gru/gru_cell/strided_slice_4/stack_1?
$gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_4/stack_2?
gru/gru_cell/strided_slice_4StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru/gru_cell/strided_slice_4?
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_1?
"gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2$
"gru/gru_cell/strided_slice_5/stack?
$gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_5/stack_1?
$gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_5/stack_2?
gru/gru_cell/strided_slice_5StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_5/stack:output:0-gru/gru_cell/strided_slice_5/stack_1:output:0-gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru/gru_cell/strided_slice_5?
gru/gru_cell/BiasAdd_2BiasAddgru/gru_cell/MatMul_2:product:0%gru/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_2?
gru/gru_cell/mulMulgru/zeros:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul?
gru/gru_cell/mul_1Mulgru/zeros:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_1?
gru/gru_cell/mul_2Mulgru/zeros:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_2?
gru/gru_cell/ReadVariableOp_4ReadVariableOp&gru_gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_4?
"gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_6/stack?
$gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$gru/gru_cell/strided_slice_6/stack_1?
$gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_6/stack_2?
gru/gru_cell/strided_slice_6StridedSlice%gru/gru_cell/ReadVariableOp_4:value:0+gru/gru_cell/strided_slice_6/stack:output:0-gru/gru_cell/strided_slice_6/stack_1:output:0-gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_6?
gru/gru_cell/MatMul_3MatMulgru/gru_cell/mul:z:0%gru/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_3?
gru/gru_cell/ReadVariableOp_5ReadVariableOp&gru_gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_5?
"gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru/gru_cell/strided_slice_7/stack?
$gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2&
$gru/gru_cell/strided_slice_7/stack_1?
$gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_7/stack_2?
gru/gru_cell/strided_slice_7StridedSlice%gru/gru_cell/ReadVariableOp_5:value:0+gru/gru_cell/strided_slice_7/stack:output:0-gru/gru_cell/strided_slice_7/stack_1:output:0-gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_7?
gru/gru_cell/MatMul_4MatMulgru/gru_cell/mul_1:z:0%gru/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_4?
"gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_8/stack?
$gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2&
$gru/gru_cell/strided_slice_8/stack_1?
$gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_8/stack_2?
gru/gru_cell/strided_slice_8StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_8/stack:output:0-gru/gru_cell/strided_slice_8/stack_1:output:0-gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru/gru_cell/strided_slice_8?
gru/gru_cell/BiasAdd_3BiasAddgru/gru_cell/MatMul_3:product:0%gru/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_3?
"gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2$
"gru/gru_cell/strided_slice_9/stack?
$gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2&
$gru/gru_cell/strided_slice_9/stack_1?
$gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_9/stack_2?
gru/gru_cell/strided_slice_9StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_9/stack:output:0-gru/gru_cell/strided_slice_9/stack_1:output:0-gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru/gru_cell/strided_slice_9?
gru/gru_cell/BiasAdd_4BiasAddgru/gru_cell/MatMul_4:product:0%gru/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_4?
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add?
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid?
gru/gru_cell/add_1AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_1?
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid_1?
gru/gru_cell/ReadVariableOp_6ReadVariableOp&gru_gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_6?
#gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2%
#gru/gru_cell/strided_slice_10/stack?
%gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%gru/gru_cell/strided_slice_10/stack_1?
%gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%gru/gru_cell/strided_slice_10/stack_2?
gru/gru_cell/strided_slice_10StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0,gru/gru_cell/strided_slice_10/stack:output:0.gru/gru_cell/strided_slice_10/stack_1:output:0.gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_10?
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_2:z:0&gru/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_5?
#gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2%
#gru/gru_cell/strided_slice_11/stack?
%gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%gru/gru_cell/strided_slice_11/stack_1?
%gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%gru/gru_cell/strided_slice_11/stack_2?
gru/gru_cell/strided_slice_11StridedSlicegru/gru_cell/unstack:output:1,gru/gru_cell/strided_slice_11/stack:output:0.gru/gru_cell/strided_slice_11/stack_1:output:0.gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru/gru_cell/strided_slice_11?
gru/gru_cell/BiasAdd_5BiasAddgru/gru_cell/MatMul_5:product:0&gru/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_5?
gru/gru_cell/mul_3Mulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_3?
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_2y
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Tanh?
gru/gru_cell/mul_4Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_4m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/sub/x?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/sub?
gru/gru_cell/mul_5Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_5?
gru/gru_cell/add_3AddV2gru/gru_cell/mul_4:z:0gru/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_3?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_1_resource&gru_gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*!
bodyR
gru_while_body_549141*!
condR
gru_while_cond_549140*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_3/stack?
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1?
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2?
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru/strided_slice_3?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru/transpose_1n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime?
layer_normalization/ShapeShapegru/strided_slice_3:output:0*
T0*
_output_shapes
:2
layer_normalization/Shape?
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack?
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1?
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/x?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul?
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stack?
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1?
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1|
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_1/x?
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1?
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0?
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shape?
layer_normalization/ReshapeReshapegru/strided_slice_3:output:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization/Reshape{
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization/Const?
layer_normalization/Fill/dimsPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2
layer_normalization/Fill/dims?
layer_normalization/FillFill&layer_normalization/Fill/dims:output:0"layer_normalization/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/Fill
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization/Const_1?
layer_normalization/Fill_1/dimsPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/Fill_1/dims?
layer_normalization/Fill_1Fill(layer_normalization/Fill_1/dims:output:0$layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/Fill_1}
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_2}
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_3?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/Fill:output:0#layer_normalization/Fill_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2&
$layer_normalization/FusedBatchNormV3?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*(
_output_shapes
:??????????2
layer_normalization/Reshape_1?
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization/mul_2/ReadVariableOp?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization/mul_2?
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&layer_normalization/add/ReadVariableOp?
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization/add?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMullayer_normalization/add:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Sigmoid?
IdentityIdentitydense/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_6
^gru/while'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12>
gru/gru_cell/ReadVariableOp_2gru/gru_cell/ReadVariableOp_22>
gru/gru_cell/ReadVariableOp_3gru/gru_cell/ReadVariableOp_32>
gru/gru_cell/ReadVariableOp_4gru/gru_cell/ReadVariableOp_42>
gru/gru_cell/ReadVariableOp_5gru/gru_cell/ReadVariableOp_52>
gru/gru_cell/ReadVariableOp_6gru/gru_cell/ReadVariableOp_62
	gru/while	gru/while2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
while_body_547964
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
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
while/gru_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
while/gru_cell/dropout/Const?
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul?
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shape?
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??u25
3while/gru_cell/dropout/random_uniform/RandomUniform?
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%while/gru_cell/dropout/GreaterEqual/y?
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2%
#while/gru_cell/dropout/GreaterEqual?
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout/Cast?
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul_1?
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_1/Const?
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Mul?
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape?
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_1/random_uniform/RandomUniform?
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell/dropout_1/GreaterEqual/y?
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_1/GreaterEqual?
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Cast?
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_1/Mul_1?
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_2/Const?
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Mul?
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape?
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_2/random_uniform/RandomUniform?
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell/dropout_2/GreaterEqual/y?
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_2/GreaterEqual?
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Cast?
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_2/Mul_1?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/mulMulwhile_placeholder_2 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mulwhile_placeholder_2"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mulwhile_placeholder_2"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0(while/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_3Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_3?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Tanh?
while/gru_cell/mul_4Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_4q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_5Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_5?
while/gru_cell/add_3AddV2while/gru_cell/mul_4:z:0while/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
?__inference_gru_layer_call_and_return_conditional_losses_547803

inputs
gru_cell_547727
gru_cell_547729
gru_cell_547731
identity?? gru_cell/StatefulPartitionedCall?whileD
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
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_547727gru_cell_547729gru_cell_547731*
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
GPU2*4,5J 8? *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_5473622"
 gru_cell/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_547727gru_cell_547729gru_cell_547731*
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
while_body_547739*
condR
while_cond_547738*9
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
IdentityIdentitystrided_slice_3:output:0!^gru_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????	
 
_user_specified_nameinputs
?
{
&__inference_dense_layer_call_fn_550665

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
GPU2*4,5J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5485042
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
?
?
$__inference_gru_layer_call_fn_550594

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
GPU2*4,5J 8? *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_5484052
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
$__inference_gru_layer_call_fn_549982
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
GPU2*4,5J 8? *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_5478032
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
?
?
A__inference_model_layer_call_and_return_conditional_losses_548521
input_1

gru_548428

gru_548430

gru_548432
layer_normalization_548488
layer_normalization_548490
dense_548515
dense_548517
identity??dense/StatefulPartitionedCall?gru/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCallinput_1
gru_548428
gru_548430
gru_548432*
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
GPU2*4,5J 8? *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_5481342
gru/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0layer_normalization_548488layer_normalization_548490*
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
GPU2*4,5J 8? *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_5484772-
+layer_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_548515dense_548517*
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
GPU2*4,5J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5485042
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_1
?
?
4__inference_layer_normalization_layer_call_fn_550645

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
GPU2*4,5J 8? *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_5484772
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
??
?

model_gru_while_body_5469230
,model_gru_while_model_gru_while_loop_counter6
2model_gru_while_model_gru_while_maximum_iterations
model_gru_while_placeholder!
model_gru_while_placeholder_1!
model_gru_while_placeholder_2/
+model_gru_while_model_gru_strided_slice_1_0k
gmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_06
2model_gru_while_gru_cell_readvariableop_resource_08
4model_gru_while_gru_cell_readvariableop_1_resource_08
4model_gru_while_gru_cell_readvariableop_4_resource_0
model_gru_while_identity
model_gru_while_identity_1
model_gru_while_identity_2
model_gru_while_identity_3
model_gru_while_identity_4-
)model_gru_while_model_gru_strided_slice_1i
emodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor4
0model_gru_while_gru_cell_readvariableop_resource6
2model_gru_while_gru_cell_readvariableop_1_resource6
2model_gru_while_gru_cell_readvariableop_4_resource??'model/gru/while/gru_cell/ReadVariableOp?)model/gru/while/gru_cell/ReadVariableOp_1?)model/gru/while/gru_cell/ReadVariableOp_2?)model/gru/while/gru_cell/ReadVariableOp_3?)model/gru/while/gru_cell/ReadVariableOp_4?)model/gru/while/gru_cell/ReadVariableOp_5?)model/gru/while/gru_cell/ReadVariableOp_6?
Amodel/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2C
Amodel/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
3model/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemgmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0model_gru_while_placeholderJmodel/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype025
3model/gru/while/TensorArrayV2Read/TensorListGetItem?
(model/gru/while/gru_cell/ones_like/ShapeShapemodel_gru_while_placeholder_2*
T0*
_output_shapes
:2*
(model/gru/while/gru_cell/ones_like/Shape?
(model/gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(model/gru/while/gru_cell/ones_like/Const?
"model/gru/while/gru_cell/ones_likeFill1model/gru/while/gru_cell/ones_like/Shape:output:01model/gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2$
"model/gru/while/gru_cell/ones_like?
'model/gru/while/gru_cell/ReadVariableOpReadVariableOp2model_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'model/gru/while/gru_cell/ReadVariableOp?
 model/gru/while/gru_cell/unstackUnpack/model/gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 model/gru/while/gru_cell/unstack?
)model/gru/while/gru_cell/ReadVariableOp_1ReadVariableOp4model_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02+
)model/gru/while/gru_cell/ReadVariableOp_1?
,model/gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,model/gru/while/gru_cell/strided_slice/stack?
.model/gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   20
.model/gru/while/gru_cell/strided_slice/stack_1?
.model/gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model/gru/while/gru_cell/strided_slice/stack_2?
&model/gru/while/gru_cell/strided_sliceStridedSlice1model/gru/while/gru_cell/ReadVariableOp_1:value:05model/gru/while/gru_cell/strided_slice/stack:output:07model/gru/while/gru_cell/strided_slice/stack_1:output:07model/gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2(
&model/gru/while/gru_cell/strided_slice?
model/gru/while/gru_cell/MatMulMatMul:model/gru/while/TensorArrayV2Read/TensorListGetItem:item:0/model/gru/while/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2!
model/gru/while/gru_cell/MatMul?
)model/gru/while/gru_cell/ReadVariableOp_2ReadVariableOp4model_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02+
)model/gru/while/gru_cell/ReadVariableOp_2?
.model/gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   20
.model/gru/while/gru_cell/strided_slice_1/stack?
0model/gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  22
0model/gru/while/gru_cell/strided_slice_1/stack_1?
0model/gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model/gru/while/gru_cell/strided_slice_1/stack_2?
(model/gru/while/gru_cell/strided_slice_1StridedSlice1model/gru/while/gru_cell/ReadVariableOp_2:value:07model/gru/while/gru_cell/strided_slice_1/stack:output:09model/gru/while/gru_cell/strided_slice_1/stack_1:output:09model/gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2*
(model/gru/while/gru_cell/strided_slice_1?
!model/gru/while/gru_cell/MatMul_1MatMul:model/gru/while/TensorArrayV2Read/TensorListGetItem:item:01model/gru/while/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2#
!model/gru/while/gru_cell/MatMul_1?
)model/gru/while/gru_cell/ReadVariableOp_3ReadVariableOp4model_gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02+
)model/gru/while/gru_cell/ReadVariableOp_3?
.model/gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  20
.model/gru/while/gru_cell/strided_slice_2/stack?
0model/gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0model/gru/while/gru_cell/strided_slice_2/stack_1?
0model/gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model/gru/while/gru_cell/strided_slice_2/stack_2?
(model/gru/while/gru_cell/strided_slice_2StridedSlice1model/gru/while/gru_cell/ReadVariableOp_3:value:07model/gru/while/gru_cell/strided_slice_2/stack:output:09model/gru/while/gru_cell/strided_slice_2/stack_1:output:09model/gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2*
(model/gru/while/gru_cell/strided_slice_2?
!model/gru/while/gru_cell/MatMul_2MatMul:model/gru/while/TensorArrayV2Read/TensorListGetItem:item:01model/gru/while/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2#
!model/gru/while/gru_cell/MatMul_2?
.model/gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model/gru/while/gru_cell/strided_slice_3/stack?
0model/gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model/gru/while/gru_cell/strided_slice_3/stack_1?
0model/gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/gru/while/gru_cell/strided_slice_3/stack_2?
(model/gru/while/gru_cell/strided_slice_3StridedSlice)model/gru/while/gru_cell/unstack:output:07model/gru/while/gru_cell/strided_slice_3/stack:output:09model/gru/while/gru_cell/strided_slice_3/stack_1:output:09model/gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2*
(model/gru/while/gru_cell/strided_slice_3?
 model/gru/while/gru_cell/BiasAddBiasAdd)model/gru/while/gru_cell/MatMul:product:01model/gru/while/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2"
 model/gru/while/gru_cell/BiasAdd?
.model/gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?20
.model/gru/while/gru_cell/strided_slice_4/stack?
0model/gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model/gru/while/gru_cell/strided_slice_4/stack_1?
0model/gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/gru/while/gru_cell/strided_slice_4/stack_2?
(model/gru/while/gru_cell/strided_slice_4StridedSlice)model/gru/while/gru_cell/unstack:output:07model/gru/while/gru_cell/strided_slice_4/stack:output:09model/gru/while/gru_cell/strided_slice_4/stack_1:output:09model/gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2*
(model/gru/while/gru_cell/strided_slice_4?
"model/gru/while/gru_cell/BiasAdd_1BiasAdd+model/gru/while/gru_cell/MatMul_1:product:01model/gru/while/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2$
"model/gru/while/gru_cell/BiasAdd_1?
.model/gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?20
.model/gru/while/gru_cell/strided_slice_5/stack?
0model/gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 22
0model/gru/while/gru_cell/strided_slice_5/stack_1?
0model/gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/gru/while/gru_cell/strided_slice_5/stack_2?
(model/gru/while/gru_cell/strided_slice_5StridedSlice)model/gru/while/gru_cell/unstack:output:07model/gru/while/gru_cell/strided_slice_5/stack:output:09model/gru/while/gru_cell/strided_slice_5/stack_1:output:09model/gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2*
(model/gru/while/gru_cell/strided_slice_5?
"model/gru/while/gru_cell/BiasAdd_2BiasAdd+model/gru/while/gru_cell/MatMul_2:product:01model/gru/while/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2$
"model/gru/while/gru_cell/BiasAdd_2?
model/gru/while/gru_cell/mulMulmodel_gru_while_placeholder_2+model/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
model/gru/while/gru_cell/mul?
model/gru/while/gru_cell/mul_1Mulmodel_gru_while_placeholder_2+model/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2 
model/gru/while/gru_cell/mul_1?
model/gru/while/gru_cell/mul_2Mulmodel_gru_while_placeholder_2+model/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2 
model/gru/while/gru_cell/mul_2?
)model/gru/while/gru_cell/ReadVariableOp_4ReadVariableOp4model_gru_while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02+
)model/gru/while/gru_cell/ReadVariableOp_4?
.model/gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.model/gru/while/gru_cell/strided_slice_6/stack?
0model/gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0model/gru/while/gru_cell/strided_slice_6/stack_1?
0model/gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model/gru/while/gru_cell/strided_slice_6/stack_2?
(model/gru/while/gru_cell/strided_slice_6StridedSlice1model/gru/while/gru_cell/ReadVariableOp_4:value:07model/gru/while/gru_cell/strided_slice_6/stack:output:09model/gru/while/gru_cell/strided_slice_6/stack_1:output:09model/gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(model/gru/while/gru_cell/strided_slice_6?
!model/gru/while/gru_cell/MatMul_3MatMul model/gru/while/gru_cell/mul:z:01model/gru/while/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2#
!model/gru/while/gru_cell/MatMul_3?
)model/gru/while/gru_cell/ReadVariableOp_5ReadVariableOp4model_gru_while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02+
)model/gru/while/gru_cell/ReadVariableOp_5?
.model/gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   20
.model/gru/while/gru_cell/strided_slice_7/stack?
0model/gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  22
0model/gru/while/gru_cell/strided_slice_7/stack_1?
0model/gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model/gru/while/gru_cell/strided_slice_7/stack_2?
(model/gru/while/gru_cell/strided_slice_7StridedSlice1model/gru/while/gru_cell/ReadVariableOp_5:value:07model/gru/while/gru_cell/strided_slice_7/stack:output:09model/gru/while/gru_cell/strided_slice_7/stack_1:output:09model/gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(model/gru/while/gru_cell/strided_slice_7?
!model/gru/while/gru_cell/MatMul_4MatMul"model/gru/while/gru_cell/mul_1:z:01model/gru/while/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2#
!model/gru/while/gru_cell/MatMul_4?
.model/gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model/gru/while/gru_cell/strided_slice_8/stack?
0model/gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model/gru/while/gru_cell/strided_slice_8/stack_1?
0model/gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/gru/while/gru_cell/strided_slice_8/stack_2?
(model/gru/while/gru_cell/strided_slice_8StridedSlice)model/gru/while/gru_cell/unstack:output:17model/gru/while/gru_cell/strided_slice_8/stack:output:09model/gru/while/gru_cell/strided_slice_8/stack_1:output:09model/gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2*
(model/gru/while/gru_cell/strided_slice_8?
"model/gru/while/gru_cell/BiasAdd_3BiasAdd+model/gru/while/gru_cell/MatMul_3:product:01model/gru/while/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2$
"model/gru/while/gru_cell/BiasAdd_3?
.model/gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?20
.model/gru/while/gru_cell/strided_slice_9/stack?
0model/gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?22
0model/gru/while/gru_cell/strided_slice_9/stack_1?
0model/gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/gru/while/gru_cell/strided_slice_9/stack_2?
(model/gru/while/gru_cell/strided_slice_9StridedSlice)model/gru/while/gru_cell/unstack:output:17model/gru/while/gru_cell/strided_slice_9/stack:output:09model/gru/while/gru_cell/strided_slice_9/stack_1:output:09model/gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2*
(model/gru/while/gru_cell/strided_slice_9?
"model/gru/while/gru_cell/BiasAdd_4BiasAdd+model/gru/while/gru_cell/MatMul_4:product:01model/gru/while/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2$
"model/gru/while/gru_cell/BiasAdd_4?
model/gru/while/gru_cell/addAddV2)model/gru/while/gru_cell/BiasAdd:output:0+model/gru/while/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
model/gru/while/gru_cell/add?
 model/gru/while/gru_cell/SigmoidSigmoid model/gru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2"
 model/gru/while/gru_cell/Sigmoid?
model/gru/while/gru_cell/add_1AddV2+model/gru/while/gru_cell/BiasAdd_1:output:0+model/gru/while/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2 
model/gru/while/gru_cell/add_1?
"model/gru/while/gru_cell/Sigmoid_1Sigmoid"model/gru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"model/gru/while/gru_cell/Sigmoid_1?
)model/gru/while/gru_cell/ReadVariableOp_6ReadVariableOp4model_gru_while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02+
)model/gru/while/gru_cell/ReadVariableOp_6?
/model/gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  21
/model/gru/while/gru_cell/strided_slice_10/stack?
1model/gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1model/gru/while/gru_cell/strided_slice_10/stack_1?
1model/gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1model/gru/while/gru_cell/strided_slice_10/stack_2?
)model/gru/while/gru_cell/strided_slice_10StridedSlice1model/gru/while/gru_cell/ReadVariableOp_6:value:08model/gru/while/gru_cell/strided_slice_10/stack:output:0:model/gru/while/gru_cell/strided_slice_10/stack_1:output:0:model/gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2+
)model/gru/while/gru_cell/strided_slice_10?
!model/gru/while/gru_cell/MatMul_5MatMul"model/gru/while/gru_cell/mul_2:z:02model/gru/while/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2#
!model/gru/while/gru_cell/MatMul_5?
/model/gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?21
/model/gru/while/gru_cell/strided_slice_11/stack?
1model/gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1model/gru/while/gru_cell/strided_slice_11/stack_1?
1model/gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model/gru/while/gru_cell/strided_slice_11/stack_2?
)model/gru/while/gru_cell/strided_slice_11StridedSlice)model/gru/while/gru_cell/unstack:output:18model/gru/while/gru_cell/strided_slice_11/stack:output:0:model/gru/while/gru_cell/strided_slice_11/stack_1:output:0:model/gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2+
)model/gru/while/gru_cell/strided_slice_11?
"model/gru/while/gru_cell/BiasAdd_5BiasAdd+model/gru/while/gru_cell/MatMul_5:product:02model/gru/while/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2$
"model/gru/while/gru_cell/BiasAdd_5?
model/gru/while/gru_cell/mul_3Mul&model/gru/while/gru_cell/Sigmoid_1:y:0+model/gru/while/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2 
model/gru/while/gru_cell/mul_3?
model/gru/while/gru_cell/add_2AddV2+model/gru/while/gru_cell/BiasAdd_2:output:0"model/gru/while/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2 
model/gru/while/gru_cell/add_2?
model/gru/while/gru_cell/TanhTanh"model/gru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
model/gru/while/gru_cell/Tanh?
model/gru/while/gru_cell/mul_4Mul$model/gru/while/gru_cell/Sigmoid:y:0model_gru_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
model/gru/while/gru_cell/mul_4?
model/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
model/gru/while/gru_cell/sub/x?
model/gru/while/gru_cell/subSub'model/gru/while/gru_cell/sub/x:output:0$model/gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
model/gru/while/gru_cell/sub?
model/gru/while/gru_cell/mul_5Mul model/gru/while/gru_cell/sub:z:0!model/gru/while/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2 
model/gru/while/gru_cell/mul_5?
model/gru/while/gru_cell/add_3AddV2"model/gru/while/gru_cell/mul_4:z:0"model/gru/while/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2 
model/gru/while/gru_cell/add_3?
4model/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmodel_gru_while_placeholder_1model_gru_while_placeholder"model/gru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype026
4model/gru/while/TensorArrayV2Write/TensorListSetItemp
model/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/gru/while/add/y?
model/gru/while/addAddV2model_gru_while_placeholdermodel/gru/while/add/y:output:0*
T0*
_output_shapes
: 2
model/gru/while/addt
model/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/gru/while/add_1/y?
model/gru/while/add_1AddV2,model_gru_while_model_gru_while_loop_counter model/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model/gru/while/add_1?
model/gru/while/IdentityIdentitymodel/gru/while/add_1:z:0(^model/gru/while/gru_cell/ReadVariableOp*^model/gru/while/gru_cell/ReadVariableOp_1*^model/gru/while/gru_cell/ReadVariableOp_2*^model/gru/while/gru_cell/ReadVariableOp_3*^model/gru/while/gru_cell/ReadVariableOp_4*^model/gru/while/gru_cell/ReadVariableOp_5*^model/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
model/gru/while/Identity?
model/gru/while/Identity_1Identity2model_gru_while_model_gru_while_maximum_iterations(^model/gru/while/gru_cell/ReadVariableOp*^model/gru/while/gru_cell/ReadVariableOp_1*^model/gru/while/gru_cell/ReadVariableOp_2*^model/gru/while/gru_cell/ReadVariableOp_3*^model/gru/while/gru_cell/ReadVariableOp_4*^model/gru/while/gru_cell/ReadVariableOp_5*^model/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
model/gru/while/Identity_1?
model/gru/while/Identity_2Identitymodel/gru/while/add:z:0(^model/gru/while/gru_cell/ReadVariableOp*^model/gru/while/gru_cell/ReadVariableOp_1*^model/gru/while/gru_cell/ReadVariableOp_2*^model/gru/while/gru_cell/ReadVariableOp_3*^model/gru/while/gru_cell/ReadVariableOp_4*^model/gru/while/gru_cell/ReadVariableOp_5*^model/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
model/gru/while/Identity_2?
model/gru/while/Identity_3IdentityDmodel/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^model/gru/while/gru_cell/ReadVariableOp*^model/gru/while/gru_cell/ReadVariableOp_1*^model/gru/while/gru_cell/ReadVariableOp_2*^model/gru/while/gru_cell/ReadVariableOp_3*^model/gru/while/gru_cell/ReadVariableOp_4*^model/gru/while/gru_cell/ReadVariableOp_5*^model/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
model/gru/while/Identity_3?
model/gru/while/Identity_4Identity"model/gru/while/gru_cell/add_3:z:0(^model/gru/while/gru_cell/ReadVariableOp*^model/gru/while/gru_cell/ReadVariableOp_1*^model/gru/while/gru_cell/ReadVariableOp_2*^model/gru/while/gru_cell/ReadVariableOp_3*^model/gru/while/gru_cell/ReadVariableOp_4*^model/gru/while/gru_cell/ReadVariableOp_5*^model/gru/while/gru_cell/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
model/gru/while/Identity_4"j
2model_gru_while_gru_cell_readvariableop_1_resource4model_gru_while_gru_cell_readvariableop_1_resource_0"j
2model_gru_while_gru_cell_readvariableop_4_resource4model_gru_while_gru_cell_readvariableop_4_resource_0"f
0model_gru_while_gru_cell_readvariableop_resource2model_gru_while_gru_cell_readvariableop_resource_0"=
model_gru_while_identity!model/gru/while/Identity:output:0"A
model_gru_while_identity_1#model/gru/while/Identity_1:output:0"A
model_gru_while_identity_2#model/gru/while/Identity_2:output:0"A
model_gru_while_identity_3#model/gru/while/Identity_3:output:0"A
model_gru_while_identity_4#model/gru/while/Identity_4:output:0"X
)model_gru_while_model_gru_strided_slice_1+model_gru_while_model_gru_strided_slice_1_0"?
emodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensorgmodel_gru_while_tensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2R
'model/gru/while/gru_cell/ReadVariableOp'model/gru/while/gru_cell/ReadVariableOp2V
)model/gru/while/gru_cell/ReadVariableOp_1)model/gru/while/gru_cell/ReadVariableOp_12V
)model/gru/while/gru_cell/ReadVariableOp_2)model/gru/while/gru_cell/ReadVariableOp_22V
)model/gru/while/gru_cell/ReadVariableOp_3)model/gru/while/gru_cell/ReadVariableOp_32V
)model/gru/while/gru_cell/ReadVariableOp_4)model/gru/while/gru_cell/ReadVariableOp_42V
)model/gru/while/gru_cell/ReadVariableOp_5)model/gru/while/gru_cell/ReadVariableOp_52V
)model/gru/while/gru_cell/ReadVariableOp_6)model/gru/while/gru_cell/ReadVariableOp_6: 
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
?h
?
D__inference_gru_cell_layer_call_and_return_conditional_losses_547362

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
?
?
A__inference_model_layer_call_and_return_conditional_losses_548606

inputs

gru_548588

gru_548590

gru_548592
layer_normalization_548595
layer_normalization_548597
dense_548600
dense_548602
identity??dense/StatefulPartitionedCall?gru/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCallinputs
gru_548588
gru_548590
gru_548592*
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
GPU2*4,5J 8? *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_5484052
gru/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0layer_normalization_548595layer_normalization_548597*
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
GPU2*4,5J 8? *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_5484772-
+layer_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_548600dense_548602*
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
GPU2*4,5J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5485042
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
A__inference_model_layer_call_and_return_conditional_losses_548542
input_1

gru_548524

gru_548526

gru_548528
layer_normalization_548531
layer_normalization_548533
dense_548536
dense_548538
identity??dense/StatefulPartitionedCall?gru/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCallinput_1
gru_548524
gru_548526
gru_548528*
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
GPU2*4,5J 8? *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_5484052
gru/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0layer_normalization_548531layer_normalization_548533*
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
GPU2*4,5J 8? *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_5484772-
+layer_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_548536dense_548538*
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
GPU2*4,5J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5485042
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_1
?
?
while_cond_547738
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_547738___redundant_placeholder04
0while_while_cond_547738___redundant_placeholder14
0while_while_cond_547738___redundant_placeholder24
0while_while_cond_547738___redundant_placeholder3
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
?
?
while_cond_550130
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_550130___redundant_placeholder04
0while_while_cond_550130___redundant_placeholder14
0while_while_cond_550130___redundant_placeholder24
0while_while_cond_550130___redundant_placeholder3
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
?
while_body_549519
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
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
while/gru_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
while/gru_cell/dropout/Const?
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul?
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shape?
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???25
3while/gru_cell/dropout/random_uniform/RandomUniform?
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%while/gru_cell/dropout/GreaterEqual/y?
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2%
#while/gru_cell/dropout/GreaterEqual?
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout/Cast?
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul_1?
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_1/Const?
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Mul?
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape?
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_1/random_uniform/RandomUniform?
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell/dropout_1/GreaterEqual/y?
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_1/GreaterEqual?
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Cast?
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_1/Mul_1?
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_2/Const?
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Mul?
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape?
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_2/random_uniform/RandomUniform?
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell/dropout_2/GreaterEqual/y?
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_2/GreaterEqual?
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Cast?
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_2/Mul_1?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/mulMulwhile_placeholder_2 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mulwhile_placeholder_2"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mulwhile_placeholder_2"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0(while/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_3Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_3?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Tanh?
while/gru_cell/mul_4Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_4q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_5Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_5?
while/gru_cell/add_3AddV2while/gru_cell/mul_4:z:0while/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
?
gru_while_cond_548800$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1<
8gru_while_gru_while_cond_548800___redundant_placeholder0<
8gru_while_gru_while_cond_548800___redundant_placeholder1<
8gru_while_gru_while_cond_548800___redundant_placeholder2<
8gru_while_gru_while_cond_548800___redundant_placeholder3
gru_while_identity
?
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*A
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
?
"__inference__traced_restore_551140
file_prefix.
*assignvariableop_layer_normalization_gamma/
+assignvariableop_1_layer_normalization_beta#
assignvariableop_2_dense_kernel!
assignvariableop_3_dense_bias!
assignvariableop_4_nadam_iter#
assignvariableop_5_nadam_beta_1#
assignvariableop_6_nadam_beta_2"
assignvariableop_7_nadam_decay*
&assignvariableop_8_nadam_learning_rate+
'assignvariableop_9_nadam_momentum_cache+
'assignvariableop_10_gru_gru_cell_kernel5
1assignvariableop_11_gru_gru_cell_recurrent_kernel)
%assignvariableop_12_gru_gru_cell_bias
assignvariableop_13_total
assignvariableop_14_count&
"assignvariableop_15_true_positives&
"assignvariableop_16_true_negatives'
#assignvariableop_17_false_positives'
#assignvariableop_18_false_negatives9
5assignvariableop_19_nadam_layer_normalization_gamma_m8
4assignvariableop_20_nadam_layer_normalization_beta_m,
(assignvariableop_21_nadam_dense_kernel_m*
&assignvariableop_22_nadam_dense_bias_m3
/assignvariableop_23_nadam_gru_gru_cell_kernel_m=
9assignvariableop_24_nadam_gru_gru_cell_recurrent_kernel_m1
-assignvariableop_25_nadam_gru_gru_cell_bias_m9
5assignvariableop_26_nadam_layer_normalization_gamma_v8
4assignvariableop_27_nadam_layer_normalization_beta_v,
(assignvariableop_28_nadam_dense_kernel_v*
&assignvariableop_29_nadam_dense_bias_v3
/assignvariableop_30_nadam_gru_gru_cell_kernel_v=
9assignvariableop_31_nadam_gru_gru_cell_recurrent_kernel_v1
-assignvariableop_32_nadam_gru_gru_cell_bias_v
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_10AssignVariableOp'assignvariableop_10_gru_gru_cell_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp1assignvariableop_11_gru_gru_cell_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_gru_gru_cell_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp5assignvariableop_19_nadam_layer_normalization_gamma_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp4assignvariableop_20_nadam_layer_normalization_beta_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_nadam_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_nadam_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_nadam_gru_gru_cell_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp9assignvariableop_24_nadam_gru_gru_cell_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp-assignvariableop_25_nadam_gru_gru_cell_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp5assignvariableop_26_nadam_layer_normalization_gamma_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp4assignvariableop_27_nadam_layer_normalization_beta_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_nadam_dense_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp&assignvariableop_29_nadam_dense_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_nadam_gru_gru_cell_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp9assignvariableop_31_nadam_gru_gru_cell_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp-assignvariableop_32_nadam_gru_gru_cell_bias_vIdentity_32:output:0"/device:CPU:0*
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
??
?
while_body_550131
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
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
while/gru_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
while/gru_cell/dropout/Const?
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul?
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shape?
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???25
3while/gru_cell/dropout/random_uniform/RandomUniform?
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%while/gru_cell/dropout/GreaterEqual/y?
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2%
#while/gru_cell/dropout/GreaterEqual?
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout/Cast?
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul_1?
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_1/Const?
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Mul?
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape?
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_1/random_uniform/RandomUniform?
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell/dropout_1/GreaterEqual/y?
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_1/GreaterEqual?
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Cast?
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_1/Mul_1?
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
while/gru_cell/dropout_2/Const?
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Mul?
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape?
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_2/random_uniform/RandomUniform?
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'while/gru_cell/dropout_2/GreaterEqual/y?
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_2/GreaterEqual?
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Cast?
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_2/Mul_1?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0'while/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_2?
while/gru_cell/mulMulwhile_placeholder_2 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mulwhile_placeholder_2"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mulwhile_placeholder_2"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul:z:0'while/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_2:z:0(while/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_3Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_3?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_2
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/Tanh?
while/gru_cell/mul_4Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_4q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/sub?
while/gru_cell/mul_5Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_5?
while/gru_cell/add_3AddV2while/gru_cell/mul_4:z:0while/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
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
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 
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
?
?__inference_gru_layer_call_and_return_conditional_losses_549689
inputs_0$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileF
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
strided_slice_2r
gru_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout/Const?
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shape?
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??92/
-gru_cell/dropout/random_uniform/RandomUniform?
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
gru_cell/dropout/GreaterEqual/y?
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/GreaterEqual?
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout/Cast?
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_1/Const?
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shape?
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_1/random_uniform/RandomUniform?
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell/dropout_1/GreaterEqual/y?
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_1/GreaterEqual?
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_1/Cast?
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru_cell/dropout_2/Const?
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shape?
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_2/random_uniform/RandomUniform?
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!gru_cell/dropout_2/GreaterEqual/y?
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_2/GreaterEqual?
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_2/Cast?
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_2?
gru_cell/mulMulzeros:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulzeros:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulzeros:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul:z:0!gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0"gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_5?
gru_cell/mul_3Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_3?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Tanh?
gru_cell/mul_4Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_4e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub
gru_cell/mul_5Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_5?
gru_cell/add_3AddV2gru_cell/mul_4:z:0gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_549519*
condR
while_cond_549518*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:^ Z
4
_output_shapes"
 :??????????????????	
"
_user_specified_name
inputs/0
?
?
gru_while_cond_549140$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1<
8gru_while_gru_while_cond_549140___redundant_placeholder0<
8gru_while_gru_while_cond_549140___redundant_placeholder1<
8gru_while_gru_while_cond_549140___redundant_placeholder2<
8gru_while_gru_while_cond_549140___redundant_placeholder3
gru_while_identity
?
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*A
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
A__inference_dense_layer_call_and_return_conditional_losses_550656

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
??
?
A__inference_model_layer_call_and_return_conditional_losses_549016

inputs(
$gru_gru_cell_readvariableop_resource*
&gru_gru_cell_readvariableop_1_resource*
&gru_gru_cell_readvariableop_4_resource5
1layer_normalization_mul_2_readvariableop_resource3
/layer_normalization_add_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?gru/gru_cell/ReadVariableOp?gru/gru_cell/ReadVariableOp_1?gru/gru_cell/ReadVariableOp_2?gru/gru_cell/ReadVariableOp_3?gru/gru_cell/ReadVariableOp_4?gru/gru_cell/ReadVariableOp_5?gru/gru_cell/ReadVariableOp_6?	gru/while?&layer_normalization/add/ReadVariableOp?(layer_normalization/mul_2/ReadVariableOpL
	gru/ShapeShapeinputs*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slicee
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessk
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/packed/1?
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const?
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transposeinputsgru/transpose/perm:output:0*
T0*+
_output_shapes
:?????????	2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
gru/strided_slice_2~
gru/gru_cell/ones_like/ShapeShapegru/zeros:output:0*
T0*
_output_shapes
:2
gru/gru_cell/ones_like/Shape?
gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/ones_like/Const?
gru/gru_cell/ones_likeFill%gru/gru_cell/ones_like/Shape:output:0%gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/ones_like}
gru/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout/Const?
gru/gru_cell/dropout/MulMulgru/gru_cell/ones_like:output:0#gru/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout/Mul?
gru/gru_cell/dropout/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout/Shape?
1gru/gru_cell/dropout/random_uniform/RandomUniformRandomUniform#gru/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??N23
1gru/gru_cell/dropout/random_uniform/RandomUniform?
#gru/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#gru/gru_cell/dropout/GreaterEqual/y?
!gru/gru_cell/dropout/GreaterEqualGreaterEqual:gru/gru_cell/dropout/random_uniform/RandomUniform:output:0,gru/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru/gru_cell/dropout/GreaterEqual?
gru/gru_cell/dropout/CastCast%gru/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru/gru_cell/dropout/Cast?
gru/gru_cell/dropout/Mul_1Mulgru/gru_cell/dropout/Mul:z:0gru/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout/Mul_1?
gru/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_1/Const?
gru/gru_cell/dropout_1/MulMulgru/gru_cell/ones_like:output:0%gru/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout_1/Mul?
gru/gru_cell/dropout_1/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_1/Shape?
3gru/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???25
3gru/gru_cell/dropout_1/random_uniform/RandomUniform?
%gru/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%gru/gru_cell/dropout_1/GreaterEqual/y?
#gru/gru_cell/dropout_1/GreaterEqualGreaterEqual<gru/gru_cell/dropout_1/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2%
#gru/gru_cell/dropout_1/GreaterEqual?
gru/gru_cell/dropout_1/CastCast'gru/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru/gru_cell/dropout_1/Cast?
gru/gru_cell/dropout_1/Mul_1Mulgru/gru_cell/dropout_1/Mul:z:0gru/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout_1/Mul_1?
gru/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gru/gru_cell/dropout_2/Const?
gru/gru_cell/dropout_2/MulMulgru/gru_cell/ones_like:output:0%gru/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout_2/Mul?
gru/gru_cell/dropout_2/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_2/Shape?
3gru/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2־?25
3gru/gru_cell/dropout_2/random_uniform/RandomUniform?
%gru/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%gru/gru_cell/dropout_2/GreaterEqual/y?
#gru/gru_cell/dropout_2/GreaterEqualGreaterEqual<gru/gru_cell/dropout_2/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2%
#gru/gru_cell/dropout_2/GreaterEqual?
gru/gru_cell/dropout_2/CastCast'gru/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru/gru_cell/dropout_2/Cast?
gru/gru_cell/dropout_2/Mul_1Mulgru/gru_cell/dropout_2/Mul:z:0gru/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout_2/Mul_1?
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru/gru_cell/ReadVariableOp?
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/gru_cell/unstack?
gru/gru_cell/ReadVariableOp_1ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru/gru_cell/ReadVariableOp_1?
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru/gru_cell/strided_slice/stack?
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru/gru_cell/strided_slice/stack_1?
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru/gru_cell/strided_slice/stack_2?
gru/gru_cell/strided_sliceStridedSlice%gru/gru_cell/ReadVariableOp_1:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice?
gru/gru_cell/MatMulMatMulgru/strided_slice_2:output:0#gru/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul?
gru/gru_cell/ReadVariableOp_2ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru/gru_cell/ReadVariableOp_2?
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru/gru_cell/strided_slice_1/stack?
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2&
$gru/gru_cell/strided_slice_1/stack_1?
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_1/stack_2?
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1?
gru/gru_cell/MatMul_1MatMulgru/strided_slice_2:output:0%gru/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_1?
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru/gru_cell/ReadVariableOp_3?
"gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2$
"gru/gru_cell/strided_slice_2/stack?
$gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_2/stack_1?
$gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_2/stack_2?
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2?
gru/gru_cell/MatMul_2MatMulgru/strided_slice_2:output:0%gru/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_2?
"gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_3/stack?
$gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2&
$gru/gru_cell/strided_slice_3/stack_1?
$gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_2?
gru/gru_cell/strided_slice_3StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru/gru_cell/strided_slice_3?
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd?
"gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2$
"gru/gru_cell/strided_slice_4/stack?
$gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2&
$gru/gru_cell/strided_slice_4/stack_1?
$gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_4/stack_2?
gru/gru_cell/strided_slice_4StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru/gru_cell/strided_slice_4?
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_1?
"gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2$
"gru/gru_cell/strided_slice_5/stack?
$gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_5/stack_1?
$gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_5/stack_2?
gru/gru_cell/strided_slice_5StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_5/stack:output:0-gru/gru_cell/strided_slice_5/stack_1:output:0-gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru/gru_cell/strided_slice_5?
gru/gru_cell/BiasAdd_2BiasAddgru/gru_cell/MatMul_2:product:0%gru/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_2?
gru/gru_cell/mulMulgru/zeros:output:0gru/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul?
gru/gru_cell/mul_1Mulgru/zeros:output:0 gru/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_1?
gru/gru_cell/mul_2Mulgru/zeros:output:0 gru/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_2?
gru/gru_cell/ReadVariableOp_4ReadVariableOp&gru_gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_4?
"gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_6/stack?
$gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2&
$gru/gru_cell/strided_slice_6/stack_1?
$gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_6/stack_2?
gru/gru_cell/strided_slice_6StridedSlice%gru/gru_cell/ReadVariableOp_4:value:0+gru/gru_cell/strided_slice_6/stack:output:0-gru/gru_cell/strided_slice_6/stack_1:output:0-gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_6?
gru/gru_cell/MatMul_3MatMulgru/gru_cell/mul:z:0%gru/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_3?
gru/gru_cell/ReadVariableOp_5ReadVariableOp&gru_gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_5?
"gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2$
"gru/gru_cell/strided_slice_7/stack?
$gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2&
$gru/gru_cell/strided_slice_7/stack_1?
$gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_7/stack_2?
gru/gru_cell/strided_slice_7StridedSlice%gru/gru_cell/ReadVariableOp_5:value:0+gru/gru_cell/strided_slice_7/stack:output:0-gru/gru_cell/strided_slice_7/stack_1:output:0-gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_7?
gru/gru_cell/MatMul_4MatMulgru/gru_cell/mul_1:z:0%gru/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_4?
"gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_8/stack?
$gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2&
$gru/gru_cell/strided_slice_8/stack_1?
$gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_8/stack_2?
gru/gru_cell/strided_slice_8StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_8/stack:output:0-gru/gru_cell/strided_slice_8/stack_1:output:0-gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru/gru_cell/strided_slice_8?
gru/gru_cell/BiasAdd_3BiasAddgru/gru_cell/MatMul_3:product:0%gru/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_3?
"gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2$
"gru/gru_cell/strided_slice_9/stack?
$gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2&
$gru/gru_cell/strided_slice_9/stack_1?
$gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_9/stack_2?
gru/gru_cell/strided_slice_9StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_9/stack:output:0-gru/gru_cell/strided_slice_9/stack_1:output:0-gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru/gru_cell/strided_slice_9?
gru/gru_cell/BiasAdd_4BiasAddgru/gru_cell/MatMul_4:product:0%gru/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_4?
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add?
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid?
gru/gru_cell/add_1AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_1?
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Sigmoid_1?
gru/gru_cell/ReadVariableOp_6ReadVariableOp&gru_gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_6?
#gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2%
#gru/gru_cell/strided_slice_10/stack?
%gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%gru/gru_cell/strided_slice_10/stack_1?
%gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%gru/gru_cell/strided_slice_10/stack_2?
gru/gru_cell/strided_slice_10StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0,gru/gru_cell/strided_slice_10/stack:output:0.gru/gru_cell/strided_slice_10/stack_1:output:0.gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_10?
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_2:z:0&gru/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/MatMul_5?
#gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2%
#gru/gru_cell/strided_slice_11/stack?
%gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%gru/gru_cell/strided_slice_11/stack_1?
%gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%gru/gru_cell/strided_slice_11/stack_2?
gru/gru_cell/strided_slice_11StridedSlicegru/gru_cell/unstack:output:1,gru/gru_cell/strided_slice_11/stack:output:0.gru/gru_cell/strided_slice_11/stack_1:output:0.gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru/gru_cell/strided_slice_11?
gru/gru_cell/BiasAdd_5BiasAddgru/gru_cell/MatMul_5:product:0&gru/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/BiasAdd_5?
gru/gru_cell/mul_3Mulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_3?
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_2y
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/Tanh?
gru/gru_cell/mul_4Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_4m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/sub/x?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/sub?
gru/gru_cell/mul_5Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_5?
gru/gru_cell/add_3AddV2gru/gru_cell/mul_4:z:0gru/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/add_3?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_1_resource&gru_gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*!
bodyR
gru_while_body_548801*!
condR
gru_while_cond_548800*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_3/stack?
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1?
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2?
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru/strided_slice_3?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru/transpose_1n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime?
layer_normalization/ShapeShapegru/strided_slice_3:output:0*
T0*
_output_shapes
:2
layer_normalization/Shape?
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack?
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1?
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/x?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul?
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stack?
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1?
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1|
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_1/x?
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1?
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0?
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shape?
layer_normalization/ReshapeReshapegru/strided_slice_3:output:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization/Reshape{
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization/Const?
layer_normalization/Fill/dimsPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2
layer_normalization/Fill/dims?
layer_normalization/FillFill&layer_normalization/Fill/dims:output:0"layer_normalization/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/Fill
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization/Const_1?
layer_normalization/Fill_1/dimsPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/Fill_1/dims?
layer_normalization/Fill_1Fill(layer_normalization/Fill_1/dims:output:0$layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/Fill_1}
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_2}
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_3?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/Fill:output:0#layer_normalization/Fill_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2&
$layer_normalization/FusedBatchNormV3?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*(
_output_shapes
:??????????2
layer_normalization/Reshape_1?
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization/mul_2/ReadVariableOp?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization/mul_2?
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&layer_normalization/add/ReadVariableOp?
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization/add?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMullayer_normalization/add:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Sigmoid?
IdentityIdentitydense/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_6
^gru/while'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12>
gru/gru_cell/ReadVariableOp_2gru/gru_cell/ReadVariableOp_22>
gru/gru_cell/ReadVariableOp_3gru/gru_cell/ReadVariableOp_32>
gru/gru_cell/ReadVariableOp_4gru/gru_cell/ReadVariableOp_42>
gru/gru_cell/ReadVariableOp_5gru/gru_cell/ReadVariableOp_52>
gru/gru_cell/ReadVariableOp_6gru/gru_cell/ReadVariableOp_62
	gru/while	gru/while2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
?__inference_gru_layer_call_and_return_conditional_losses_550572

inputs$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileD
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
strided_slice_2r
gru_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_like?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_2?
gru_cell/mulMulzeros:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulzeros:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulzeros:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul:z:0!gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0"gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_5?
gru_cell/mul_3Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_3?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Tanh?
gru_cell/mul_4Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_4e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub
gru_cell/mul_5Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_5?
gru_cell/add_3AddV2gru_cell/mul_4:z:0gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_550426*
condR
while_cond_550425*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
$__inference_gru_layer_call_fn_549971
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
GPU2*4,5J 8? *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_5476852
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
?<
?
?__inference_gru_layer_call_and_return_conditional_losses_547685

inputs
gru_cell_547609
gru_cell_547611
gru_cell_547613
identity?? gru_cell/StatefulPartitionedCall?whileD
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
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_547609gru_cell_547611gru_cell_547613*
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
GPU2*4,5J 8? *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_5472662"
 gru_cell/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_547609gru_cell_547611gru_cell_547613*
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
while_body_547621*
condR
while_cond_547620*9
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
IdentityIdentitystrided_slice_3:output:0!^gru_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????	
 
_user_specified_nameinputs
?
?
$__inference_gru_layer_call_fn_550583

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
GPU2*4,5J 8? *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_5481342
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
?
?
A__inference_model_layer_call_and_return_conditional_losses_548566

inputs

gru_548548

gru_548550

gru_548552
layer_normalization_548555
layer_normalization_548557
dense_548560
dense_548562
identity??dense/StatefulPartitionedCall?gru/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCallinputs
gru_548548
gru_548550
gru_548552*
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
GPU2*4,5J 8? *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_5481342
gru/StatefulPartitionedCall?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0layer_normalization_548555layer_normalization_548557*
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
GPU2*4,5J 8? *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_5484772-
+layer_normalization/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_548560dense_548562*
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
GPU2*4,5J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5485042
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs
??
?
?__inference_gru_layer_call_and_return_conditional_losses_549960
inputs_0$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileF
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
strided_slice_2r
gru_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_like?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulstrided_slice_2:output:0gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulstrided_slice_2:output:0!gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulstrided_slice_2:output:0!gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_2?
gru_cell/mulMulzeros:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulzeros:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulzeros:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul:z:0!gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/addt
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_1z
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_2:z:0"gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/BiasAdd_5?
gru_cell/mul_3Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_3?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_2m
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/Tanh?
gru_cell/mul_4Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_4e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/sub
gru_cell/mul_5Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_5?
gru_cell/add_3AddV2gru_cell/mul_4:z:0gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
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
while_body_549814*
condR
while_cond_549813*9
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
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????	:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:^ Z
4
_output_shapes"
 :??????????????????	
"
_user_specified_name
inputs/0
?"
?
O__inference_layer_normalization_layer_call_and_return_conditional_losses_550636

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
while_body_547739
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_547761_0
while_gru_cell_547763_0
while_gru_cell_547765_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_547761
while_gru_cell_547763
while_gru_cell_547765??&while/gru_cell/StatefulPartitionedCall?
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
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_547761_0while_gru_cell_547763_0while_gru_cell_547765_0*
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
GPU2*4,5J 8? *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_5473622(
&while/gru_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4"0
while_gru_cell_547761while_gru_cell_547761_0"0
while_gru_cell_547763while_gru_cell_547763_0"0
while_gru_cell_547765while_gru_cell_547765_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 
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
&__inference_model_layer_call_fn_549351

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
GPU2*4,5J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5485662
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
?
?
while_cond_547620
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_547620___redundant_placeholder04
0while_while_cond_547620___redundant_placeholder14
0while_while_cond_547620___redundant_placeholder24
0while_while_cond_547620___redundant_placeholder3
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
)__inference_gru_cell_layer_call_fn_550909

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
GPU2*4,5J 8? *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_5473622
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
??
?
!__inference__wrapped_model_547114
input_1.
*model_gru_gru_cell_readvariableop_resource0
,model_gru_gru_cell_readvariableop_1_resource0
,model_gru_gru_cell_readvariableop_4_resource;
7model_layer_normalization_mul_2_readvariableop_resource9
5model_layer_normalization_add_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource
identity??"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?!model/gru/gru_cell/ReadVariableOp?#model/gru/gru_cell/ReadVariableOp_1?#model/gru/gru_cell/ReadVariableOp_2?#model/gru/gru_cell/ReadVariableOp_3?#model/gru/gru_cell/ReadVariableOp_4?#model/gru/gru_cell/ReadVariableOp_5?#model/gru/gru_cell/ReadVariableOp_6?model/gru/while?,model/layer_normalization/add/ReadVariableOp?.model/layer_normalization/mul_2/ReadVariableOpY
model/gru/ShapeShapeinput_1*
T0*
_output_shapes
:2
model/gru/Shape?
model/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
model/gru/strided_slice/stack?
model/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
model/gru/strided_slice/stack_1?
model/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
model/gru/strided_slice/stack_2?
model/gru/strided_sliceStridedSlicemodel/gru/Shape:output:0&model/gru/strided_slice/stack:output:0(model/gru/strided_slice/stack_1:output:0(model/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/gru/strided_sliceq
model/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/gru/zeros/mul/y?
model/gru/zeros/mulMul model/gru/strided_slice:output:0model/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model/gru/zeros/muls
model/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/gru/zeros/Less/y?
model/gru/zeros/LessLessmodel/gru/zeros/mul:z:0model/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model/gru/zeros/Lessw
model/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
model/gru/zeros/packed/1?
model/gru/zeros/packedPack model/gru/strided_slice:output:0!model/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/gru/zeros/packeds
model/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/gru/zeros/Const?
model/gru/zerosFillmodel/gru/zeros/packed:output:0model/gru/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
model/gru/zeros?
model/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/gru/transpose/perm?
model/gru/transpose	Transposeinput_1!model/gru/transpose/perm:output:0*
T0*+
_output_shapes
:?????????	2
model/gru/transposem
model/gru/Shape_1Shapemodel/gru/transpose:y:0*
T0*
_output_shapes
:2
model/gru/Shape_1?
model/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
model/gru/strided_slice_1/stack?
!model/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_1/stack_1?
!model/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_1/stack_2?
model/gru/strided_slice_1StridedSlicemodel/gru/Shape_1:output:0(model/gru/strided_slice_1/stack:output:0*model/gru/strided_slice_1/stack_1:output:0*model/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/gru/strided_slice_1?
%model/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%model/gru/TensorArrayV2/element_shape?
model/gru/TensorArrayV2TensorListReserve.model/gru/TensorArrayV2/element_shape:output:0"model/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/gru/TensorArrayV2?
?model/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2A
?model/gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
1model/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/gru/transpose:y:0Hmodel/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1model/gru/TensorArrayUnstack/TensorListFromTensor?
model/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
model/gru/strided_slice_2/stack?
!model/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_2/stack_1?
!model/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_2/stack_2?
model/gru/strided_slice_2StridedSlicemodel/gru/transpose:y:0(model/gru/strided_slice_2/stack:output:0*model/gru/strided_slice_2/stack_1:output:0*model/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*
shrink_axis_mask2
model/gru/strided_slice_2?
"model/gru/gru_cell/ones_like/ShapeShapemodel/gru/zeros:output:0*
T0*
_output_shapes
:2$
"model/gru/gru_cell/ones_like/Shape?
"model/gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"model/gru/gru_cell/ones_like/Const?
model/gru/gru_cell/ones_likeFill+model/gru/gru_cell/ones_like/Shape:output:0+model/gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/ones_like?
!model/gru/gru_cell/ReadVariableOpReadVariableOp*model_gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!model/gru/gru_cell/ReadVariableOp?
model/gru/gru_cell/unstackUnpack)model/gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
model/gru/gru_cell/unstack?
#model/gru/gru_cell/ReadVariableOp_1ReadVariableOp,model_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02%
#model/gru/gru_cell/ReadVariableOp_1?
&model/gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&model/gru/gru_cell/strided_slice/stack?
(model/gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(model/gru/gru_cell/strided_slice/stack_1?
(model/gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(model/gru/gru_cell/strided_slice/stack_2?
 model/gru/gru_cell/strided_sliceStridedSlice+model/gru/gru_cell/ReadVariableOp_1:value:0/model/gru/gru_cell/strided_slice/stack:output:01model/gru/gru_cell/strided_slice/stack_1:output:01model/gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 model/gru/gru_cell/strided_slice?
model/gru/gru_cell/MatMulMatMul"model/gru/strided_slice_2:output:0)model/gru/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/MatMul?
#model/gru/gru_cell/ReadVariableOp_2ReadVariableOp,model_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02%
#model/gru/gru_cell/ReadVariableOp_2?
(model/gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(model/gru/gru_cell/strided_slice_1/stack?
*model/gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2,
*model/gru/gru_cell/strided_slice_1/stack_1?
*model/gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*model/gru/gru_cell/strided_slice_1/stack_2?
"model/gru/gru_cell/strided_slice_1StridedSlice+model/gru/gru_cell/ReadVariableOp_2:value:01model/gru/gru_cell/strided_slice_1/stack:output:03model/gru/gru_cell/strided_slice_1/stack_1:output:03model/gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2$
"model/gru/gru_cell/strided_slice_1?
model/gru/gru_cell/MatMul_1MatMul"model/gru/strided_slice_2:output:0+model/gru/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/MatMul_1?
#model/gru/gru_cell/ReadVariableOp_3ReadVariableOp,model_gru_gru_cell_readvariableop_1_resource*
_output_shapes
:		?*
dtype02%
#model/gru/gru_cell/ReadVariableOp_3?
(model/gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(model/gru/gru_cell/strided_slice_2/stack?
*model/gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*model/gru/gru_cell/strided_slice_2/stack_1?
*model/gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*model/gru/gru_cell/strided_slice_2/stack_2?
"model/gru/gru_cell/strided_slice_2StridedSlice+model/gru/gru_cell/ReadVariableOp_3:value:01model/gru/gru_cell/strided_slice_2/stack:output:03model/gru/gru_cell/strided_slice_2/stack_1:output:03model/gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2$
"model/gru/gru_cell/strided_slice_2?
model/gru/gru_cell/MatMul_2MatMul"model/gru/strided_slice_2:output:0+model/gru/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/MatMul_2?
(model/gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model/gru/gru_cell/strided_slice_3/stack?
*model/gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*model/gru/gru_cell/strided_slice_3/stack_1?
*model/gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/gru/gru_cell/strided_slice_3/stack_2?
"model/gru/gru_cell/strided_slice_3StridedSlice#model/gru/gru_cell/unstack:output:01model/gru/gru_cell/strided_slice_3/stack:output:03model/gru/gru_cell/strided_slice_3/stack_1:output:03model/gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2$
"model/gru/gru_cell/strided_slice_3?
model/gru/gru_cell/BiasAddBiasAdd#model/gru/gru_cell/MatMul:product:0+model/gru/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/BiasAdd?
(model/gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2*
(model/gru/gru_cell/strided_slice_4/stack?
*model/gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*model/gru/gru_cell/strided_slice_4/stack_1?
*model/gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/gru/gru_cell/strided_slice_4/stack_2?
"model/gru/gru_cell/strided_slice_4StridedSlice#model/gru/gru_cell/unstack:output:01model/gru/gru_cell/strided_slice_4/stack:output:03model/gru/gru_cell/strided_slice_4/stack_1:output:03model/gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2$
"model/gru/gru_cell/strided_slice_4?
model/gru/gru_cell/BiasAdd_1BiasAdd%model/gru/gru_cell/MatMul_1:product:0+model/gru/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/BiasAdd_1?
(model/gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2*
(model/gru/gru_cell/strided_slice_5/stack?
*model/gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*model/gru/gru_cell/strided_slice_5/stack_1?
*model/gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/gru/gru_cell/strided_slice_5/stack_2?
"model/gru/gru_cell/strided_slice_5StridedSlice#model/gru/gru_cell/unstack:output:01model/gru/gru_cell/strided_slice_5/stack:output:03model/gru/gru_cell/strided_slice_5/stack_1:output:03model/gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2$
"model/gru/gru_cell/strided_slice_5?
model/gru/gru_cell/BiasAdd_2BiasAdd%model/gru/gru_cell/MatMul_2:product:0+model/gru/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/BiasAdd_2?
model/gru/gru_cell/mulMulmodel/gru/zeros:output:0%model/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/mul?
model/gru/gru_cell/mul_1Mulmodel/gru/zeros:output:0%model/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/mul_1?
model/gru/gru_cell/mul_2Mulmodel/gru/zeros:output:0%model/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/mul_2?
#model/gru/gru_cell/ReadVariableOp_4ReadVariableOp,model_gru_gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02%
#model/gru/gru_cell/ReadVariableOp_4?
(model/gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(model/gru/gru_cell/strided_slice_6/stack?
*model/gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2,
*model/gru/gru_cell/strided_slice_6/stack_1?
*model/gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*model/gru/gru_cell/strided_slice_6/stack_2?
"model/gru/gru_cell/strided_slice_6StridedSlice+model/gru/gru_cell/ReadVariableOp_4:value:01model/gru/gru_cell/strided_slice_6/stack:output:03model/gru/gru_cell/strided_slice_6/stack_1:output:03model/gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"model/gru/gru_cell/strided_slice_6?
model/gru/gru_cell/MatMul_3MatMulmodel/gru/gru_cell/mul:z:0+model/gru/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/MatMul_3?
#model/gru/gru_cell/ReadVariableOp_5ReadVariableOp,model_gru_gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02%
#model/gru/gru_cell/ReadVariableOp_5?
(model/gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(model/gru/gru_cell/strided_slice_7/stack?
*model/gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2,
*model/gru/gru_cell/strided_slice_7/stack_1?
*model/gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*model/gru/gru_cell/strided_slice_7/stack_2?
"model/gru/gru_cell/strided_slice_7StridedSlice+model/gru/gru_cell/ReadVariableOp_5:value:01model/gru/gru_cell/strided_slice_7/stack:output:03model/gru/gru_cell/strided_slice_7/stack_1:output:03model/gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"model/gru/gru_cell/strided_slice_7?
model/gru/gru_cell/MatMul_4MatMulmodel/gru/gru_cell/mul_1:z:0+model/gru/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/MatMul_4?
(model/gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model/gru/gru_cell/strided_slice_8/stack?
*model/gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*model/gru/gru_cell/strided_slice_8/stack_1?
*model/gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/gru/gru_cell/strided_slice_8/stack_2?
"model/gru/gru_cell/strided_slice_8StridedSlice#model/gru/gru_cell/unstack:output:11model/gru/gru_cell/strided_slice_8/stack:output:03model/gru/gru_cell/strided_slice_8/stack_1:output:03model/gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2$
"model/gru/gru_cell/strided_slice_8?
model/gru/gru_cell/BiasAdd_3BiasAdd%model/gru/gru_cell/MatMul_3:product:0+model/gru/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/BiasAdd_3?
(model/gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2*
(model/gru/gru_cell/strided_slice_9/stack?
*model/gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*model/gru/gru_cell/strided_slice_9/stack_1?
*model/gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/gru/gru_cell/strided_slice_9/stack_2?
"model/gru/gru_cell/strided_slice_9StridedSlice#model/gru/gru_cell/unstack:output:11model/gru/gru_cell/strided_slice_9/stack:output:03model/gru/gru_cell/strided_slice_9/stack_1:output:03model/gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2$
"model/gru/gru_cell/strided_slice_9?
model/gru/gru_cell/BiasAdd_4BiasAdd%model/gru/gru_cell/MatMul_4:product:0+model/gru/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/BiasAdd_4?
model/gru/gru_cell/addAddV2#model/gru/gru_cell/BiasAdd:output:0%model/gru/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/add?
model/gru/gru_cell/SigmoidSigmoidmodel/gru/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/Sigmoid?
model/gru/gru_cell/add_1AddV2%model/gru/gru_cell/BiasAdd_1:output:0%model/gru/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/add_1?
model/gru/gru_cell/Sigmoid_1Sigmoidmodel/gru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/Sigmoid_1?
#model/gru/gru_cell/ReadVariableOp_6ReadVariableOp,model_gru_gru_cell_readvariableop_4_resource* 
_output_shapes
:
??*
dtype02%
#model/gru/gru_cell/ReadVariableOp_6?
)model/gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2+
)model/gru/gru_cell/strided_slice_10/stack?
+model/gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+model/gru/gru_cell/strided_slice_10/stack_1?
+model/gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+model/gru/gru_cell/strided_slice_10/stack_2?
#model/gru/gru_cell/strided_slice_10StridedSlice+model/gru/gru_cell/ReadVariableOp_6:value:02model/gru/gru_cell/strided_slice_10/stack:output:04model/gru/gru_cell/strided_slice_10/stack_1:output:04model/gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2%
#model/gru/gru_cell/strided_slice_10?
model/gru/gru_cell/MatMul_5MatMulmodel/gru/gru_cell/mul_2:z:0,model/gru/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/MatMul_5?
)model/gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2+
)model/gru/gru_cell/strided_slice_11/stack?
+model/gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+model/gru/gru_cell/strided_slice_11/stack_1?
+model/gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+model/gru/gru_cell/strided_slice_11/stack_2?
#model/gru/gru_cell/strided_slice_11StridedSlice#model/gru/gru_cell/unstack:output:12model/gru/gru_cell/strided_slice_11/stack:output:04model/gru/gru_cell/strided_slice_11/stack_1:output:04model/gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2%
#model/gru/gru_cell/strided_slice_11?
model/gru/gru_cell/BiasAdd_5BiasAdd%model/gru/gru_cell/MatMul_5:product:0,model/gru/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/BiasAdd_5?
model/gru/gru_cell/mul_3Mul model/gru/gru_cell/Sigmoid_1:y:0%model/gru/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/mul_3?
model/gru/gru_cell/add_2AddV2%model/gru/gru_cell/BiasAdd_2:output:0model/gru/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/add_2?
model/gru/gru_cell/TanhTanhmodel/gru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/Tanh?
model/gru/gru_cell/mul_4Mulmodel/gru/gru_cell/Sigmoid:y:0model/gru/zeros:output:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/mul_4y
model/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
model/gru/gru_cell/sub/x?
model/gru/gru_cell/subSub!model/gru/gru_cell/sub/x:output:0model/gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/sub?
model/gru/gru_cell/mul_5Mulmodel/gru/gru_cell/sub:z:0model/gru/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/mul_5?
model/gru/gru_cell/add_3AddV2model/gru/gru_cell/mul_4:z:0model/gru/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
model/gru/gru_cell/add_3?
'model/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2)
'model/gru/TensorArrayV2_1/element_shape?
model/gru/TensorArrayV2_1TensorListReserve0model/gru/TensorArrayV2_1/element_shape:output:0"model/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/gru/TensorArrayV2_1b
model/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model/gru/time?
"model/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model/gru/while/maximum_iterations~
model/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
model/gru/while/loop_counter?
model/gru/whileWhile%model/gru/while/loop_counter:output:0+model/gru/while/maximum_iterations:output:0model/gru/time:output:0"model/gru/TensorArrayV2_1:handle:0model/gru/zeros:output:0"model/gru/strided_slice_1:output:0Amodel/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0*model_gru_gru_cell_readvariableop_resource,model_gru_gru_cell_readvariableop_1_resource,model_gru_gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*'
bodyR
model_gru_while_body_546923*'
condR
model_gru_while_cond_546922*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
model/gru/while?
:model/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2<
:model/gru/TensorArrayV2Stack/TensorListStack/element_shape?
,model/gru/TensorArrayV2Stack/TensorListStackTensorListStackmodel/gru/while:output:3Cmodel/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02.
,model/gru/TensorArrayV2Stack/TensorListStack?
model/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2!
model/gru/strided_slice_3/stack?
!model/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model/gru/strided_slice_3/stack_1?
!model/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_3/stack_2?
model/gru/strided_slice_3StridedSlice5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0(model/gru/strided_slice_3/stack:output:0*model/gru/strided_slice_3/stack_1:output:0*model/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
model/gru/strided_slice_3?
model/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/gru/transpose_1/perm?
model/gru/transpose_1	Transpose5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0#model/gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
model/gru/transpose_1z
model/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model/gru/runtime?
model/layer_normalization/ShapeShape"model/gru/strided_slice_3:output:0*
T0*
_output_shapes
:2!
model/layer_normalization/Shape?
-model/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-model/layer_normalization/strided_slice/stack?
/model/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/layer_normalization/strided_slice/stack_1?
/model/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/layer_normalization/strided_slice/stack_2?
'model/layer_normalization/strided_sliceStridedSlice(model/layer_normalization/Shape:output:06model/layer_normalization/strided_slice/stack:output:08model/layer_normalization/strided_slice/stack_1:output:08model/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/layer_normalization/strided_slice?
model/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2!
model/layer_normalization/mul/x?
model/layer_normalization/mulMul(model/layer_normalization/mul/x:output:00model/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
model/layer_normalization/mul?
/model/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/model/layer_normalization/strided_slice_1/stack?
1model/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model/layer_normalization/strided_slice_1/stack_1?
1model/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model/layer_normalization/strided_slice_1/stack_2?
)model/layer_normalization/strided_slice_1StridedSlice(model/layer_normalization/Shape:output:08model/layer_normalization/strided_slice_1/stack:output:0:model/layer_normalization/strided_slice_1/stack_1:output:0:model/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model/layer_normalization/strided_slice_1?
!model/layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2#
!model/layer_normalization/mul_1/x?
model/layer_normalization/mul_1Mul*model/layer_normalization/mul_1/x:output:02model/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2!
model/layer_normalization/mul_1?
)model/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2+
)model/layer_normalization/Reshape/shape/0?
)model/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)model/layer_normalization/Reshape/shape/3?
'model/layer_normalization/Reshape/shapePack2model/layer_normalization/Reshape/shape/0:output:0!model/layer_normalization/mul:z:0#model/layer_normalization/mul_1:z:02model/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'model/layer_normalization/Reshape/shape?
!model/layer_normalization/ReshapeReshape"model/gru/strided_slice_3:output:00model/layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2#
!model/layer_normalization/Reshape?
model/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
model/layer_normalization/Const?
#model/layer_normalization/Fill/dimsPack!model/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2%
#model/layer_normalization/Fill/dims?
model/layer_normalization/FillFill,model/layer_normalization/Fill/dims:output:0(model/layer_normalization/Const:output:0*
T0*#
_output_shapes
:?????????2 
model/layer_normalization/Fill?
!model/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2#
!model/layer_normalization/Const_1?
%model/layer_normalization/Fill_1/dimsPack!model/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2'
%model/layer_normalization/Fill_1/dims?
 model/layer_normalization/Fill_1Fill.model/layer_normalization/Fill_1/dims:output:0*model/layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:?????????2"
 model/layer_normalization/Fill_1?
!model/layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2#
!model/layer_normalization/Const_2?
!model/layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2#
!model/layer_normalization/Const_3?
*model/layer_normalization/FusedBatchNormV3FusedBatchNormV3*model/layer_normalization/Reshape:output:0'model/layer_normalization/Fill:output:0)model/layer_normalization/Fill_1:output:0*model/layer_normalization/Const_2:output:0*model/layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2,
*model/layer_normalization/FusedBatchNormV3?
#model/layer_normalization/Reshape_1Reshape.model/layer_normalization/FusedBatchNormV3:y:0(model/layer_normalization/Shape:output:0*
T0*(
_output_shapes
:??????????2%
#model/layer_normalization/Reshape_1?
.model/layer_normalization/mul_2/ReadVariableOpReadVariableOp7model_layer_normalization_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model/layer_normalization/mul_2/ReadVariableOp?
model/layer_normalization/mul_2Mul,model/layer_normalization/Reshape_1:output:06model/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
model/layer_normalization/mul_2?
,model/layer_normalization/add/ReadVariableOpReadVariableOp5model_layer_normalization_add_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,model/layer_normalization/add/ReadVariableOp?
model/layer_normalization/addAddV2#model/layer_normalization/mul_2:z:04model/layer_normalization/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/layer_normalization/add?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul!model/layer_normalization/add:z:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/BiasAdd?
model/dense/SigmoidSigmoidmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense/Sigmoid?
IdentityIdentitymodel/dense/Sigmoid:y:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp"^model/gru/gru_cell/ReadVariableOp$^model/gru/gru_cell/ReadVariableOp_1$^model/gru/gru_cell/ReadVariableOp_2$^model/gru/gru_cell/ReadVariableOp_3$^model/gru/gru_cell/ReadVariableOp_4$^model/gru/gru_cell/ReadVariableOp_5$^model/gru/gru_cell/ReadVariableOp_6^model/gru/while-^model/layer_normalization/add/ReadVariableOp/^model/layer_normalization/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????	:::::::2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2F
!model/gru/gru_cell/ReadVariableOp!model/gru/gru_cell/ReadVariableOp2J
#model/gru/gru_cell/ReadVariableOp_1#model/gru/gru_cell/ReadVariableOp_12J
#model/gru/gru_cell/ReadVariableOp_2#model/gru/gru_cell/ReadVariableOp_22J
#model/gru/gru_cell/ReadVariableOp_3#model/gru/gru_cell/ReadVariableOp_32J
#model/gru/gru_cell/ReadVariableOp_4#model/gru/gru_cell/ReadVariableOp_42J
#model/gru/gru_cell/ReadVariableOp_5#model/gru/gru_cell/ReadVariableOp_52J
#model/gru/gru_cell/ReadVariableOp_6#model/gru/gru_cell/ReadVariableOp_62"
model/gru/whilemodel/gru/while2\
,model/layer_normalization/add/ReadVariableOp,model/layer_normalization/add/ReadVariableOp2`
.model/layer_normalization/mul_2/ReadVariableOp.model/layer_normalization/mul_2/ReadVariableOp:T P
+
_output_shapes
:?????????	
!
_user_specified_name	input_1
??
?
gru_while_body_548801$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_00
,gru_while_gru_cell_readvariableop_resource_02
.gru_while_gru_cell_readvariableop_1_resource_02
.gru_while_gru_cell_readvariableop_4_resource_0
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor.
*gru_while_gru_cell_readvariableop_resource0
,gru_while_gru_cell_readvariableop_1_resource0
,gru_while_gru_cell_readvariableop_4_resource??!gru/while/gru_cell/ReadVariableOp?#gru/while/gru_cell/ReadVariableOp_1?#gru/while/gru_cell/ReadVariableOp_2?#gru/while/gru_cell/ReadVariableOp_3?#gru/while/gru_cell/ReadVariableOp_4?#gru/while/gru_cell/ReadVariableOp_5?#gru/while/gru_cell/ReadVariableOp_6?
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????	   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????	*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem?
"gru/while/gru_cell/ones_like/ShapeShapegru_while_placeholder_2*
T0*
_output_shapes
:2$
"gru/while/gru_cell/ones_like/Shape?
"gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru/while/gru_cell/ones_like/Const?
gru/while/gru_cell/ones_likeFill+gru/while/gru_cell/ones_like/Shape:output:0+gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/ones_like?
 gru/while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 gru/while/gru_cell/dropout/Const?
gru/while/gru_cell/dropout/MulMul%gru/while/gru_cell/ones_like:output:0)gru/while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2 
gru/while/gru_cell/dropout/Mul?
 gru/while/gru_cell/dropout/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2"
 gru/while/gru_cell/dropout/Shape?
7gru/while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform)gru/while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???29
7gru/while/gru_cell/dropout/random_uniform/RandomUniform?
)gru/while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)gru/while/gru_cell/dropout/GreaterEqual/y?
'gru/while/gru_cell/dropout/GreaterEqualGreaterEqual@gru/while/gru_cell/dropout/random_uniform/RandomUniform:output:02gru/while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'gru/while/gru_cell/dropout/GreaterEqual?
gru/while/gru_cell/dropout/CastCast+gru/while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
gru/while/gru_cell/dropout/Cast?
 gru/while/gru_cell/dropout/Mul_1Mul"gru/while/gru_cell/dropout/Mul:z:0#gru/while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 gru/while/gru_cell/dropout/Mul_1?
"gru/while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_1/Const?
 gru/while/gru_cell/dropout_1/MulMul%gru/while/gru_cell/ones_like:output:0+gru/while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2"
 gru/while/gru_cell/dropout_1/Mul?
"gru/while/gru_cell/dropout_1/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_1/Shape?
9gru/while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2;
9gru/while/gru_cell/dropout_1/random_uniform/RandomUniform?
+gru/while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+gru/while/gru_cell/dropout_1/GreaterEqual/y?
)gru/while/gru_cell/dropout_1/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_1/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)gru/while/gru_cell/dropout_1/GreaterEqual?
!gru/while/gru_cell/dropout_1/CastCast-gru/while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!gru/while/gru_cell/dropout_1/Cast?
"gru/while/gru_cell/dropout_1/Mul_1Mul$gru/while/gru_cell/dropout_1/Mul:z:0%gru/while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"gru/while/gru_cell/dropout_1/Mul_1?
"gru/while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"gru/while/gru_cell/dropout_2/Const?
 gru/while/gru_cell/dropout_2/MulMul%gru/while/gru_cell/ones_like:output:0+gru/while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2"
 gru/while/gru_cell/dropout_2/Mul?
"gru/while/gru_cell/dropout_2/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_2/Shape?
9gru/while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2;
9gru/while/gru_cell/dropout_2/random_uniform/RandomUniform?
+gru/while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+gru/while/gru_cell/dropout_2/GreaterEqual/y?
)gru/while/gru_cell/dropout_2/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_2/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)gru/while/gru_cell/dropout_2/GreaterEqual?
!gru/while/gru_cell/dropout_2/CastCast-gru/while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!gru/while/gru_cell/dropout_2/Cast?
"gru/while/gru_cell/dropout_2/Mul_1Mul$gru/while/gru_cell/dropout_2/Mul:z:0%gru/while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"gru/while/gru_cell/dropout_2/Mul_1?
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02#
!gru/while/gru_cell/ReadVariableOp?
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/while/gru_cell/unstack?
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1?
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stack?
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(gru/while/gru_cell/strided_slice/stack_1?
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2?
 gru/while/gru_cell/strided_sliceStridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_slice?
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0)gru/while/gru_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul?
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2?
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(gru/while/gru_cell/strided_slice_1/stack?
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2,
*gru/while/gru_cell/strided_slice_1/stack_1?
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_1/stack_2?
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1?
gru/while/gru_cell/MatMul_1MatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_1?
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0*
_output_shapes
:		?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3?
(gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(gru/while/gru_cell/strided_slice_2/stack?
*gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_2/stack_1?
*gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_2/stack_2?
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:		?*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2?
gru/while/gru_cell/MatMul_2MatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_2?
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stack?
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*gru/while/gru_cell/strided_slice_3/stack_1?
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2?
"gru/while/gru_cell/strided_slice_3StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2$
"gru/while/gru_cell/strided_slice_3?
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd?
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?2*
(gru/while/gru_cell/strided_slice_4/stack?
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*gru/while/gru_cell/strided_slice_4/stack_1?
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2?
"gru/while/gru_cell/strided_slice_4StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2$
"gru/while/gru_cell/strided_slice_4?
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_1?
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:?2*
(gru/while/gru_cell/strided_slice_5/stack?
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1?
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2?
"gru/while/gru_cell/strided_slice_5StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2$
"gru/while/gru_cell/strided_slice_5?
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_2?
gru/while/gru_cell/mulMulgru_while_placeholder_2$gru/while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul?
gru/while/gru_cell/mul_1Mulgru_while_placeholder_2&gru/while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_1?
gru/while/gru_cell/mul_2Mulgru_while_placeholder_2&gru/while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_2?
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4?
(gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_6/stack?
*gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2,
*gru/while/gru_cell/strided_slice_6/stack_1?
*gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_6/stack_2?
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6?
gru/while/gru_cell/MatMul_3MatMulgru/while/gru_cell/mul:z:0+gru/while/gru_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_3?
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5?
(gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2*
(gru/while/gru_cell/strided_slice_7/stack?
*gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2,
*gru/while/gru_cell/strided_slice_7/stack_1?
*gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_7/stack_2?
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7?
gru/while/gru_cell/MatMul_4MatMulgru/while/gru_cell/mul_1:z:0+gru/while/gru_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_4?
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_8/stack?
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*gru/while/gru_cell/strided_slice_8/stack_1?
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_8/stack_2?
"gru/while/gru_cell/strided_slice_8StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*

begin_mask2$
"gru/while/gru_cell/strided_slice_8?
gru/while/gru_cell/BiasAdd_3BiasAdd%gru/while/gru_cell/MatMul_3:product:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_3?
(gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:?2*
(gru/while/gru_cell/strided_slice_9/stack?
*gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?2,
*gru/while/gru_cell/strided_slice_9/stack_1?
*gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_9/stack_2?
"gru/while/gru_cell/strided_slice_9StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_9/stack:output:03gru/while/gru_cell/strided_slice_9/stack_1:output:03gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?2$
"gru/while/gru_cell/strided_slice_9?
gru/while/gru_cell/BiasAdd_4BiasAdd%gru/while/gru_cell/MatMul_4:product:0+gru/while/gru_cell/strided_slice_9:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_4?
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add?
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid?
gru/while/gru_cell/add_1AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_1?
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Sigmoid_1?
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6?
)gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2+
)gru/while/gru_cell/strided_slice_10/stack?
+gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+gru/while/gru_cell/strided_slice_10/stack_1?
+gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+gru/while/gru_cell/strided_slice_10/stack_2?
#gru/while/gru_cell/strided_slice_10StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:02gru/while/gru_cell/strided_slice_10/stack:output:04gru/while/gru_cell/strided_slice_10/stack_1:output:04gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2%
#gru/while/gru_cell/strided_slice_10?
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_2:z:0,gru/while/gru_cell/strided_slice_10:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/MatMul_5?
)gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:?2+
)gru/while/gru_cell/strided_slice_11/stack?
+gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gru/while/gru_cell/strided_slice_11/stack_1?
+gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gru/while/gru_cell/strided_slice_11/stack_2?
#gru/while/gru_cell/strided_slice_11StridedSlice#gru/while/gru_cell/unstack:output:12gru/while/gru_cell/strided_slice_11/stack:output:04gru/while/gru_cell/strided_slice_11/stack_1:output:04gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes	
:?*
end_mask2%
#gru/while/gru_cell/strided_slice_11?
gru/while/gru_cell/BiasAdd_5BiasAdd%gru/while/gru_cell/MatMul_5:product:0,gru/while/gru_cell/strided_slice_11:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/BiasAdd_5?
gru/while/gru_cell/mul_3Mul gru/while/gru_cell/Sigmoid_1:y:0%gru/while/gru_cell/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_3?
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_2:output:0gru/while/gru_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_2?
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/Tanh?
gru/while/gru_cell/mul_4Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_4y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/while/gru_cell/sub/x?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/sub?
gru/while/gru_cell/mul_5Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_5?
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_4:z:0gru/while/gru_cell/mul_5:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/add_3?
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y?
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1?
gru/while/IdentityIdentitygru/while/add_1:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity?
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1?
gru/while/Identity_2Identitygru/while/add:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2?
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3?
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*(
_output_shapes
:??????????2
gru/while/Identity_4"^
,gru_while_gru_cell_readvariableop_1_resource.gru_while_gru_cell_readvariableop_1_resource_0"^
,gru_while_gru_cell_readvariableop_4_resource.gru_while_gru_cell_readvariableop_4_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"?
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp2J
#gru/while/gru_cell/ReadVariableOp_1#gru/while/gru_cell/ReadVariableOp_12J
#gru/while/gru_cell/ReadVariableOp_2#gru/while/gru_cell/ReadVariableOp_22J
#gru/while/gru_cell/ReadVariableOp_3#gru/while/gru_cell/ReadVariableOp_32J
#gru/while/gru_cell/ReadVariableOp_4#gru/while/gru_cell/ReadVariableOp_42J
#gru/while/gru_cell/ReadVariableOp_5#gru/while/gru_cell/ReadVariableOp_52J
#gru/while/gru_cell/ReadVariableOp_6#gru/while/gru_cell/ReadVariableOp_6: 
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
?
model_gru_while_cond_5469220
,model_gru_while_model_gru_while_loop_counter6
2model_gru_while_model_gru_while_maximum_iterations
model_gru_while_placeholder!
model_gru_while_placeholder_1!
model_gru_while_placeholder_22
.model_gru_while_less_model_gru_strided_slice_1H
Dmodel_gru_while_model_gru_while_cond_546922___redundant_placeholder0H
Dmodel_gru_while_model_gru_while_cond_546922___redundant_placeholder1H
Dmodel_gru_while_model_gru_while_cond_546922___redundant_placeholder2H
Dmodel_gru_while_model_gru_while_cond_546922___redundant_placeholder3
model_gru_while_identity
?
model/gru/while/LessLessmodel_gru_while_placeholder.model_gru_while_less_model_gru_strided_slice_1*
T0*
_output_shapes
: 2
model/gru/while/Less{
model/gru/while/IdentityIdentitymodel/gru/while/Less:z:0*
T0
*
_output_shapes
: 2
model/gru/while/Identity"=
model_gru_while_identity!model/gru/while/Identity:output:0*A
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
)__inference_gru_cell_layer_call_fn_550895

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
GPU2*4,5J 8? *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_5472662
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
states/0"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????	9
dense0
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
trainable_variables
	variables
regularization_losses
		keras_api


signatures
*_&call_and_return_all_conditional_losses
`_default_save_signature
a__call__"?J
_tf_keras_network?J{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}, "name": "gru", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization", "inbound_nodes": [[["gru", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}, "name": "gru", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization", "inbound_nodes": [[["gru", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 5.000000413701855e-10, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?

_tf_keras_rnn_layer?	{"class_name": "GRU", "name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 9]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 9]}}
?
axis
	gamma
beta
trainable_variables
	variables
regularization_losses
	keras_api
*d&call_and_return_all_conditional_losses
e__call__"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*f&call_and_return_all_conditional_losses
g__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
iter

beta_1

 beta_2
	!decay
"learning_rate
#momentum_cachemQmRmSmT$mU%mV&mWvXvYvZv[$v\%v]&v^"
	optimizer
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
 "
trackable_list_wrapper
?
'metrics

(layers
)layer_metrics
trainable_variables
*non_trainable_variables
	variables
+layer_regularization_losses
regularization_losses
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
,trainable_variables
-	variables
.regularization_losses
/	keras_api
*i&call_and_return_all_conditional_losses
j__call__"?
_tf_keras_layer?{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.5, "implementation": 1, "reset_after": true}}
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
 "
trackable_list_wrapper
?

0states
1metrics

2layers
3layer_metrics
trainable_variables
4non_trainable_variables
	variables
5layer_regularization_losses
regularization_losses
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&?2layer_normalization/gamma
':%?2layer_normalization/beta
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
6metrics

7layers
8layer_metrics
trainable_variables
9non_trainable_variables
	variables
:layer_regularization_losses
regularization_losses
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
:	?2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;metrics

<layers
=layer_metrics
trainable_variables
>non_trainable_variables
	variables
?layer_regularization_losses
regularization_losses
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
&:$		?2gru/gru_cell/kernel
1:/
??2gru/gru_cell/recurrent_kernel
$:"	?2gru/gru_cell/bias
.
@0
A1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
?
Bmetrics

Clayers
Dlayer_metrics
,trainable_variables
Enon_trainable_variables
-	variables
Flayer_regularization_losses
.regularization_losses
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
.:,?2!Nadam/layer_normalization/gamma/m
-:+?2 Nadam/layer_normalization/beta/m
%:#	?2Nadam/dense/kernel/m
:2Nadam/dense/bias/m
,:*		?2Nadam/gru/gru_cell/kernel/m
7:5
??2%Nadam/gru/gru_cell/recurrent_kernel/m
*:(	?2Nadam/gru/gru_cell/bias/m
.:,?2!Nadam/layer_normalization/gamma/v
-:+?2 Nadam/layer_normalization/beta/v
%:#	?2Nadam/dense/kernel/v
:2Nadam/dense/bias/v
,:*		?2Nadam/gru/gru_cell/kernel/v
7:5
??2%Nadam/gru/gru_cell/recurrent_kernel/v
*:(	?2Nadam/gru/gru_cell/bias/v
?2?
A__inference_model_layer_call_and_return_conditional_losses_549332
A__inference_model_layer_call_and_return_conditional_losses_548542
A__inference_model_layer_call_and_return_conditional_losses_549016
A__inference_model_layer_call_and_return_conditional_losses_548521?
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
!__inference__wrapped_model_547114?
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
input_1?????????	
?2?
&__inference_model_layer_call_fn_548623
&__inference_model_layer_call_fn_548583
&__inference_model_layer_call_fn_549351
&__inference_model_layer_call_fn_549370?
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
?__inference_gru_layer_call_and_return_conditional_losses_549689
?__inference_gru_layer_call_and_return_conditional_losses_549960
?__inference_gru_layer_call_and_return_conditional_losses_550572
?__inference_gru_layer_call_and_return_conditional_losses_550301?
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
$__inference_gru_layer_call_fn_550583
$__inference_gru_layer_call_fn_549982
$__inference_gru_layer_call_fn_549971
$__inference_gru_layer_call_fn_550594?
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
O__inference_layer_normalization_layer_call_and_return_conditional_losses_550636?
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
4__inference_layer_normalization_layer_call_fn_550645?
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
A__inference_dense_layer_call_and_return_conditional_losses_550656?
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
&__inference_dense_layer_call_fn_550665?
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
$__inference_signature_wrapper_548652input_1"?
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
D__inference_gru_cell_layer_call_and_return_conditional_losses_550785
D__inference_gru_cell_layer_call_and_return_conditional_losses_550881?
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
)__inference_gru_cell_layer_call_fn_550909
)__inference_gru_cell_layer_call_fn_550895?
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
!__inference__wrapped_model_547114n&$%4?1
*?'
%?"
input_1?????????	
? "-?*
(
dense?
dense??????????
A__inference_dense_layer_call_and_return_conditional_losses_550656]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_dense_layer_call_fn_550665P0?-
&?#
!?
inputs??????????
? "???????????
D__inference_gru_cell_layer_call_and_return_conditional_losses_550785?&$%]?Z
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
D__inference_gru_cell_layer_call_and_return_conditional_losses_550881?&$%]?Z
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
)__inference_gru_cell_layer_call_fn_550895?&$%]?Z
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
)__inference_gru_cell_layer_call_fn_550909?&$%]?Z
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
?__inference_gru_layer_call_and_return_conditional_losses_549689~&$%O?L
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
?__inference_gru_layer_call_and_return_conditional_losses_549960~&$%O?L
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
?__inference_gru_layer_call_and_return_conditional_losses_550301n&$%??<
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
?__inference_gru_layer_call_and_return_conditional_losses_550572n&$%??<
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
$__inference_gru_layer_call_fn_549971q&$%O?L
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
$__inference_gru_layer_call_fn_549982q&$%O?L
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
$__inference_gru_layer_call_fn_550583a&$%??<
5?2
$?!
inputs?????????	

 
p

 
? "????????????
$__inference_gru_layer_call_fn_550594a&$%??<
5?2
$?!
inputs?????????	

 
p 

 
? "????????????
O__inference_layer_normalization_layer_call_and_return_conditional_losses_550636^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
4__inference_layer_normalization_layer_call_fn_550645Q0?-
&?#
!?
inputs??????????
? "????????????
A__inference_model_layer_call_and_return_conditional_losses_548521n&$%<?9
2?/
%?"
input_1?????????	
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_548542n&$%<?9
2?/
%?"
input_1?????????	
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_549016m&$%;?8
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
A__inference_model_layer_call_and_return_conditional_losses_549332m&$%;?8
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
&__inference_model_layer_call_fn_548583a&$%<?9
2?/
%?"
input_1?????????	
p

 
? "???????????
&__inference_model_layer_call_fn_548623a&$%<?9
2?/
%?"
input_1?????????	
p 

 
? "???????????
&__inference_model_layer_call_fn_549351`&$%;?8
1?.
$?!
inputs?????????	
p

 
? "???????????
&__inference_model_layer_call_fn_549370`&$%;?8
1?.
$?!
inputs?????????	
p 

 
? "???????????
$__inference_signature_wrapper_548652y&$%??<
? 
5?2
0
input_1%?"
input_1?????????	"-?*
(
dense?
dense?????????