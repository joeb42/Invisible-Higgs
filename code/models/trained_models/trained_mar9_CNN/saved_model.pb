??
??
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?}?*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
?}?*
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
??*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
??*
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
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
??*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:?*
dtype0
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
??*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:?*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	?	*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:	*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
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
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
z
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_positives_1
s
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes
:*
dtype0
x
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_2
q
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes
:*
dtype0
z
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:*
dtype0
?
Nadam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameNadam/conv2d_2/kernel/m
?
+Nadam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_2/kernel/m*&
_output_shapes
: *
dtype0
?
Nadam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_2/bias/m
{
)Nadam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_2/bias/m*
_output_shapes
: *
dtype0
?
Nadam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameNadam/conv2d_3/kernel/m
?
+Nadam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_3/kernel/m*&
_output_shapes
:  *
dtype0
?
Nadam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_3/bias/m
{
)Nadam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_3/bias/m*
_output_shapes
: *
dtype0
?
Nadam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?}?*'
shared_nameNadam/dense_4/kernel/m
?
*Nadam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_4/kernel/m* 
_output_shapes
:
?}?*
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
??*'
shared_nameNadam/dense_5/kernel/m
?
*Nadam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_5/kernel/m* 
_output_shapes
:
??*
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
dtype0*
shape:
??*'
shared_nameNadam/dense_7/kernel/m
?
*Nadam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_7/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameNadam/dense_7/bias/m
z
(Nadam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_7/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameNadam/dense_8/kernel/m
?
*Nadam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_8/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameNadam/dense_8/bias/m
z
(Nadam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_8/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*'
shared_nameNadam/dense_9/kernel/m
?
*Nadam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_9/kernel/m*
_output_shapes
:	?	*
dtype0
?
Nadam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameNadam/dense_9/bias/m
y
(Nadam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_9/bias/m*
_output_shapes
:	*
dtype0
?
Nadam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameNadam/conv2d_2/kernel/v
?
+Nadam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_2/kernel/v*&
_output_shapes
: *
dtype0
?
Nadam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_2/bias/v
{
)Nadam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_2/bias/v*
_output_shapes
: *
dtype0
?
Nadam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameNadam/conv2d_3/kernel/v
?
+Nadam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_3/kernel/v*&
_output_shapes
:  *
dtype0
?
Nadam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_3/bias/v
{
)Nadam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_3/bias/v*
_output_shapes
: *
dtype0
?
Nadam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?}?*'
shared_nameNadam/dense_4/kernel/v
?
*Nadam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_4/kernel/v* 
_output_shapes
:
?}?*
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
??*'
shared_nameNadam/dense_5/kernel/v
?
*Nadam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_5/kernel/v* 
_output_shapes
:
??*
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
dtype0*
shape:
??*'
shared_nameNadam/dense_7/kernel/v
?
*Nadam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_7/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameNadam/dense_7/bias/v
z
(Nadam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_7/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameNadam/dense_8/kernel/v
?
*Nadam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_8/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameNadam/dense_8/bias/v
z
(Nadam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_8/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*'
shared_nameNadam/dense_9/kernel/v
?
*Nadam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_9/kernel/v*
_output_shapes
:	?	*
dtype0
?
Nadam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameNadam/dense_9/bias/v
y
(Nadam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_9/bias/v*
_output_shapes
:	*
dtype0

NoOpNoOp
?r
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?r
value?qB?q B?q
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
	optimizer

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
%
#_self_saveable_object_factories
?

kernel
bias
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
?

 kernel
!bias
#"_self_saveable_object_factories
#trainable_variables
$	variables
%regularization_losses
&	keras_api
w
#'_self_saveable_object_factories
(trainable_variables
)	variables
*regularization_losses
+	keras_api
w
#,_self_saveable_object_factories
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?

1kernel
2bias
#3_self_saveable_object_factories
4trainable_variables
5	variables
6regularization_losses
7	keras_api
w
#8_self_saveable_object_factories
9trainable_variables
:	variables
;regularization_losses
<	keras_api
?

=kernel
>bias
#?_self_saveable_object_factories
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
w
#D_self_saveable_object_factories
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
?

Ikernel
Jbias
#K_self_saveable_object_factories
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
w
#P_self_saveable_object_factories
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
?

Ukernel
Vbias
#W_self_saveable_object_factories
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
w
#\_self_saveable_object_factories
]trainable_variables
^	variables
_regularization_losses
`	keras_api
?

akernel
bbias
#c_self_saveable_object_factories
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
w
#h_self_saveable_object_factories
itrainable_variables
j	variables
kregularization_losses
l	keras_api
?

mkernel
nbias
#o_self_saveable_object_factories
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
?
titer

ubeta_1

vbeta_2
	wdecay
xlearning_rate
ymomentum_cachem?m? m?!m?1m?2m?=m?>m?Im?Jm?Um?Vm?am?bm?mm?nm?v?v? v?!v?1v?2v?=v?>v?Iv?Jv?Uv?Vv?av?bv?mv?nv?
 
 
v
0
1
 2
!3
14
25
=6
>7
I8
J9
U10
V11
a12
b13
m14
n15
v
0
1
 2
!3
14
25
=6
>7
I8
J9
U10
V11
a12
b13
m14
n15
 
?
zlayer_metrics
{non_trainable_variables
|metrics
	variables
}layer_regularization_losses

~layers
trainable_variables
regularization_losses
 
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
?
layer_metrics
?metrics
trainable_variables
	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
 
?
?layer_metrics
?metrics
#trainable_variables
$	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
%regularization_losses
 
 
 
 
?
?layer_metrics
?metrics
(trainable_variables
)	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
*regularization_losses
 
 
 
 
?
?layer_metrics
?metrics
-trainable_variables
.	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
/regularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
 
?
?layer_metrics
?metrics
4trainable_variables
5	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
6regularization_losses
 
 
 
 
?
?layer_metrics
?metrics
9trainable_variables
:	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
;regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
 
?
?layer_metrics
?metrics
@trainable_variables
A	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Bregularization_losses
 
 
 
 
?
?layer_metrics
?metrics
Etrainable_variables
F	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Gregularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

I0
J1
 
?
?layer_metrics
?metrics
Ltrainable_variables
M	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Nregularization_losses
 
 
 
 
?
?layer_metrics
?metrics
Qtrainable_variables
R	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Sregularization_losses
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1

U0
V1
 
?
?layer_metrics
?metrics
Xtrainable_variables
Y	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Zregularization_losses
 
 
 
 
?
?layer_metrics
?metrics
]trainable_variables
^	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
_regularization_losses
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1

a0
b1
 
?
?layer_metrics
?metrics
dtrainable_variables
e	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
fregularization_losses
 
 
 
 
?
?layer_metrics
?metrics
itrainable_variables
j	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
kregularization_losses
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

m0
n1
 
?
?layer_metrics
?metrics
ptrainable_variables
q	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
rregularization_losses
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
 
 
(
?0
?1
?2
?3
?4
 
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
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
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
v
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api
\
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api
\
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
ca
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
}
VARIABLE_VALUENadam/conv2d_2/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv2d_2/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv2d_3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv2d_3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
~|
VARIABLE_VALUENadam/dense_8/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_8/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_9/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_9/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv2d_2/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv2d_2/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv2d_3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv2d_3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
~|
VARIABLE_VALUENadam/dense_8/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_8/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_9/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_9/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_2Placeholder*/
_output_shapes
:?????????(2*
dtype0*$
shape:?????????(2
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*3J 8? */
f*R(
&__inference_signature_wrapper_11622279
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp+Nadam/conv2d_2/kernel/m/Read/ReadVariableOp)Nadam/conv2d_2/bias/m/Read/ReadVariableOp+Nadam/conv2d_3/kernel/m/Read/ReadVariableOp)Nadam/conv2d_3/bias/m/Read/ReadVariableOp*Nadam/dense_4/kernel/m/Read/ReadVariableOp(Nadam/dense_4/bias/m/Read/ReadVariableOp*Nadam/dense_5/kernel/m/Read/ReadVariableOp(Nadam/dense_5/bias/m/Read/ReadVariableOp*Nadam/dense_6/kernel/m/Read/ReadVariableOp(Nadam/dense_6/bias/m/Read/ReadVariableOp*Nadam/dense_7/kernel/m/Read/ReadVariableOp(Nadam/dense_7/bias/m/Read/ReadVariableOp*Nadam/dense_8/kernel/m/Read/ReadVariableOp(Nadam/dense_8/bias/m/Read/ReadVariableOp*Nadam/dense_9/kernel/m/Read/ReadVariableOp(Nadam/dense_9/bias/m/Read/ReadVariableOp+Nadam/conv2d_2/kernel/v/Read/ReadVariableOp)Nadam/conv2d_2/bias/v/Read/ReadVariableOp+Nadam/conv2d_3/kernel/v/Read/ReadVariableOp)Nadam/conv2d_3/bias/v/Read/ReadVariableOp*Nadam/dense_4/kernel/v/Read/ReadVariableOp(Nadam/dense_4/bias/v/Read/ReadVariableOp*Nadam/dense_5/kernel/v/Read/ReadVariableOp(Nadam/dense_5/bias/v/Read/ReadVariableOp*Nadam/dense_6/kernel/v/Read/ReadVariableOp(Nadam/dense_6/bias/v/Read/ReadVariableOp*Nadam/dense_7/kernel/v/Read/ReadVariableOp(Nadam/dense_7/bias/v/Read/ReadVariableOp*Nadam/dense_8/kernel/v/Read/ReadVariableOp(Nadam/dense_8/bias/v/Read/ReadVariableOp*Nadam/dense_9/kernel/v/Read/ReadVariableOp(Nadam/dense_9/bias/v/Read/ReadVariableOpConst*O
TinH
F2D	*
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
GPU2*3J 8? **
f%R#
!__inference__traced_save_11623171
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcounttotal_1count_1true_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1false_positives_1true_positives_2false_negatives_1Nadam/conv2d_2/kernel/mNadam/conv2d_2/bias/mNadam/conv2d_3/kernel/mNadam/conv2d_3/bias/mNadam/dense_4/kernel/mNadam/dense_4/bias/mNadam/dense_5/kernel/mNadam/dense_5/bias/mNadam/dense_6/kernel/mNadam/dense_6/bias/mNadam/dense_7/kernel/mNadam/dense_7/bias/mNadam/dense_8/kernel/mNadam/dense_8/bias/mNadam/dense_9/kernel/mNadam/dense_9/bias/mNadam/conv2d_2/kernel/vNadam/conv2d_2/bias/vNadam/conv2d_3/kernel/vNadam/conv2d_3/bias/vNadam/dense_4/kernel/vNadam/dense_4/bias/vNadam/dense_5/kernel/vNadam/dense_5/bias/vNadam/dense_6/kernel/vNadam/dense_6/bias/vNadam/dense_7/kernel/vNadam/dense_7/bias/vNadam/dense_8/kernel/vNadam/dense_8/bias/vNadam/dense_9/kernel/vNadam/dense_9/bias/v*N
TinG
E2C*
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
GPU2*3J 8? *-
f(R&
$__inference__traced_restore_11623379??
?
l
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_11622797

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_uniform/max?
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??A2
random_uniform/RandomUniform?
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub?
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
random_uniform/mul?
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
random_uniforme
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
GreaterEqual/y?
GreaterEqualGreaterEqualrandom_uniform:z:0GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
CastV
mulMulinputsCast:y:0*
T0*(
_output_shapes
:??????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x^
subSubsub/x:output:0Cast:y:0*
T0*(
_output_shapes
:??????????2
subW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2	
mul_1/xc
mul_1Mulmul_1/x:output:0sub:z:0*
T0*(
_output_shapes
:??????????2
mul_1Z
addAddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
addW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2	
mul_2/xc
mul_2Mulmul_2/x:output:0add:z:0*
T0*(
_output_shapes
:??????????2
mul_2W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2	
add_1/yg
add_1AddV2	mul_2:z:0add_1/y:output:0*
T0*(
_output_shapes
:??????????2
add_1^
IdentityIdentity	add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_11622738

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_uniform/max?
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??"2
random_uniform/RandomUniform?
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub?
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
random_uniform/mul?
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
random_uniforme
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
GreaterEqual/y?
GreaterEqualGreaterEqualrandom_uniform:z:0GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
CastV
mulMulinputsCast:y:0*
T0*(
_output_shapes
:??????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x^
subSubsub/x:output:0Cast:y:0*
T0*(
_output_shapes
:??????????2
subW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2	
mul_1/xc
mul_1Mulmul_1/x:output:0sub:z:0*
T0*(
_output_shapes
:??????????2
mul_1Z
addAddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
addW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2	
mul_2/xc
mul_2Mulmul_2/x:output:0add:z:0*
T0*(
_output_shapes
:??????????2
mul_2W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2	
add_1/yg
add_1AddV2	mul_2:z:0add_1/y:output:0*
T0*(
_output_shapes
:??????????2
add_1^
IdentityIdentity	add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_11622920

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_11621825

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_5_layer_call_and_return_conditional_losses_11622705

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
?
k
2__inference_alpha_dropout_7_layer_call_fn_11622866

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_116218892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_4_layer_call_and_return_conditional_losses_11621642

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?}?*
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
:??????????}::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????}
 
_user_specified_nameinputs
?	
?
E__inference_dense_9_layer_call_and_return_conditional_losses_11622941

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

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
?
i
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_11621963

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_11622915

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_uniform/max?
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??m2
random_uniform/RandomUniform?
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub?
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
random_uniform/mul?
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
random_uniforme
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
GreaterEqual/y?
GreaterEqualGreaterEqualrandom_uniform:z:0GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
CastV
mulMulinputsCast:y:0*
T0*(
_output_shapes
:??????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x^
subSubsub/x:output:0Cast:y:0*
T0*(
_output_shapes
:??????????2
subW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2	
mul_1/xc
mul_1Mulmul_1/x:output:0sub:z:0*
T0*(
_output_shapes
:??????????2
mul_1Z
addAddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
addW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2	
mul_2/xc
mul_2Mulmul_2/x:output:0add:z:0*
T0*(
_output_shapes
:??????????2
mul_2W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2	
add_1/yg
add_1AddV2	mul_2:z:0add_1/y:output:0*
T0*(
_output_shapes
:??????????2
add_1^
IdentityIdentity	add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?I
?
C__inference_model_layer_call_and_return_conditional_losses_11622109

inputs
conv2d_2_11622061
conv2d_2_11622063
conv2d_3_11622066
conv2d_3_11622068
dense_4_11622073
dense_4_11622075
dense_5_11622079
dense_5_11622081
dense_6_11622085
dense_6_11622087
dense_7_11622091
dense_7_11622093
dense_8_11622097
dense_8_11622099
dense_9_11622103
dense_9_11622105
identity??'alpha_dropout_4/StatefulPartitionedCall?'alpha_dropout_5/StatefulPartitionedCall?'alpha_dropout_6/StatefulPartitionedCall?'alpha_dropout_7/StatefulPartitionedCall?'alpha_dropout_8/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_11622061conv2d_2_11622063*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_116215732"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_11622066conv2d_3_11622068*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_116216002"
 conv2d_3/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_116215522!
max_pooling2d_1/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_116216232
flatten_1/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_11622073dense_4_11622075*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_116216422!
dense_4/StatefulPartitionedCall?
'alpha_dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_116216822)
'alpha_dropout_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall0alpha_dropout_4/StatefulPartitionedCall:output:0dense_5_11622079dense_5_11622081*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_116217112!
dense_5/StatefulPartitionedCall?
'alpha_dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0(^alpha_dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_116217512)
'alpha_dropout_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall0alpha_dropout_5/StatefulPartitionedCall:output:0dense_6_11622085dense_6_11622087*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_116217802!
dense_6/StatefulPartitionedCall?
'alpha_dropout_6/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0(^alpha_dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_116218202)
'alpha_dropout_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall0alpha_dropout_6/StatefulPartitionedCall:output:0dense_7_11622091dense_7_11622093*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_116218492!
dense_7/StatefulPartitionedCall?
'alpha_dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0(^alpha_dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_116218892)
'alpha_dropout_7/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall0alpha_dropout_7/StatefulPartitionedCall:output:0dense_8_11622097dense_8_11622099*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_116219182!
dense_8/StatefulPartitionedCall?
'alpha_dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0(^alpha_dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_116219582)
'alpha_dropout_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall0alpha_dropout_8/StatefulPartitionedCall:output:0dense_9_11622103dense_9_11622105*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_116219872!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0(^alpha_dropout_4/StatefulPartitionedCall(^alpha_dropout_5/StatefulPartitionedCall(^alpha_dropout_6/StatefulPartitionedCall(^alpha_dropout_7/StatefulPartitionedCall(^alpha_dropout_8/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::2R
'alpha_dropout_4/StatefulPartitionedCall'alpha_dropout_4/StatefulPartitionedCall2R
'alpha_dropout_5/StatefulPartitionedCall'alpha_dropout_5/StatefulPartitionedCall2R
'alpha_dropout_6/StatefulPartitionedCall'alpha_dropout_6/StatefulPartitionedCall2R
'alpha_dropout_7/StatefulPartitionedCall'alpha_dropout_7/StatefulPartitionedCall2R
'alpha_dropout_8/StatefulPartitionedCall'alpha_dropout_8/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????(2
 
_user_specified_nameinputs
?
H
,__inference_flatten_1_layer_call_fn_11622635

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_116216232
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????}2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
l
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_11621751

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_uniform/max?
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2
random_uniform/RandomUniform?
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub?
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
random_uniform/mul?
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
random_uniforme
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
GreaterEqual/y?
GreaterEqualGreaterEqualrandom_uniform:z:0GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
CastV
mulMulinputsCast:y:0*
T0*(
_output_shapes
:??????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x^
subSubsub/x:output:0Cast:y:0*
T0*(
_output_shapes
:??????????2
subW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2	
mul_1/xc
mul_1Mulmul_1/x:output:0sub:z:0*
T0*(
_output_shapes
:??????????2
mul_1Z
addAddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
addW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2	
mul_2/xc
mul_2Mulmul_2/x:output:0add:z:0*
T0*(
_output_shapes
:??????????2
mul_2W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2	
add_1/yg
add_1AddV2	mul_2:z:0add_1/y:output:0*
T0*(
_output_shapes
:??????????2
add_1^
IdentityIdentity	add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
(__inference_model_layer_call_fn_11622547

inputs
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

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*3J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_116221092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(2
 
_user_specified_nameinputs
?
?
+__inference_conv2d_2_layer_call_fn_11622604

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
 */
_output_shapes
:?????????(2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_116215732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(2 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(2::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(2
 
_user_specified_nameinputs
?@
?
C__inference_model_layer_call_and_return_conditional_losses_11622055
input_2
conv2d_2_11622007
conv2d_2_11622009
conv2d_3_11622012
conv2d_3_11622014
dense_4_11622019
dense_4_11622021
dense_5_11622025
dense_5_11622027
dense_6_11622031
dense_6_11622033
dense_7_11622037
dense_7_11622039
dense_8_11622043
dense_8_11622045
dense_9_11622049
dense_9_11622051
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_2_11622007conv2d_2_11622009*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_116215732"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_11622012conv2d_3_11622014*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_116216002"
 conv2d_3/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_116215522!
max_pooling2d_1/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_116216232
flatten_1/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_11622019dense_4_11622021*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_116216422!
dense_4/StatefulPartitionedCall?
alpha_dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_116216872!
alpha_dropout_4/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(alpha_dropout_4/PartitionedCall:output:0dense_5_11622025dense_5_11622027*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_116217112!
dense_5/StatefulPartitionedCall?
alpha_dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_116217562!
alpha_dropout_5/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall(alpha_dropout_5/PartitionedCall:output:0dense_6_11622031dense_6_11622033*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_116217802!
dense_6/StatefulPartitionedCall?
alpha_dropout_6/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_116218252!
alpha_dropout_6/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(alpha_dropout_6/PartitionedCall:output:0dense_7_11622037dense_7_11622039*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_116218492!
dense_7/StatefulPartitionedCall?
alpha_dropout_7/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_116218942!
alpha_dropout_7/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall(alpha_dropout_7/PartitionedCall:output:0dense_8_11622043dense_8_11622045*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_116219182!
dense_8/StatefulPartitionedCall?
alpha_dropout_8/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_116219632!
alpha_dropout_8/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(alpha_dropout_8/PartitionedCall:output:0dense_9_11622049dense_9_11622051*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_116219872!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:X T
/
_output_shapes
:?????????(2
!
_user_specified_name	input_2
??
?!
$__inference__traced_restore_11623379
file_prefix$
 assignvariableop_conv2d_2_kernel$
 assignvariableop_1_conv2d_2_bias&
"assignvariableop_2_conv2d_3_kernel$
 assignvariableop_3_conv2d_3_bias%
!assignvariableop_4_dense_4_kernel#
assignvariableop_5_dense_4_bias%
!assignvariableop_6_dense_5_kernel#
assignvariableop_7_dense_5_bias%
!assignvariableop_8_dense_6_kernel#
assignvariableop_9_dense_6_bias&
"assignvariableop_10_dense_7_kernel$
 assignvariableop_11_dense_7_bias&
"assignvariableop_12_dense_8_kernel$
 assignvariableop_13_dense_8_bias&
"assignvariableop_14_dense_9_kernel$
 assignvariableop_15_dense_9_bias"
assignvariableop_16_nadam_iter$
 assignvariableop_17_nadam_beta_1$
 assignvariableop_18_nadam_beta_2#
assignvariableop_19_nadam_decay+
'assignvariableop_20_nadam_learning_rate,
(assignvariableop_21_nadam_momentum_cache
assignvariableop_22_total
assignvariableop_23_count
assignvariableop_24_total_1
assignvariableop_25_count_1&
"assignvariableop_26_true_positives&
"assignvariableop_27_true_negatives'
#assignvariableop_28_false_positives'
#assignvariableop_29_false_negatives(
$assignvariableop_30_true_positives_1)
%assignvariableop_31_false_positives_1(
$assignvariableop_32_true_positives_2)
%assignvariableop_33_false_negatives_1/
+assignvariableop_34_nadam_conv2d_2_kernel_m-
)assignvariableop_35_nadam_conv2d_2_bias_m/
+assignvariableop_36_nadam_conv2d_3_kernel_m-
)assignvariableop_37_nadam_conv2d_3_bias_m.
*assignvariableop_38_nadam_dense_4_kernel_m,
(assignvariableop_39_nadam_dense_4_bias_m.
*assignvariableop_40_nadam_dense_5_kernel_m,
(assignvariableop_41_nadam_dense_5_bias_m.
*assignvariableop_42_nadam_dense_6_kernel_m,
(assignvariableop_43_nadam_dense_6_bias_m.
*assignvariableop_44_nadam_dense_7_kernel_m,
(assignvariableop_45_nadam_dense_7_bias_m.
*assignvariableop_46_nadam_dense_8_kernel_m,
(assignvariableop_47_nadam_dense_8_bias_m.
*assignvariableop_48_nadam_dense_9_kernel_m,
(assignvariableop_49_nadam_dense_9_bias_m/
+assignvariableop_50_nadam_conv2d_2_kernel_v-
)assignvariableop_51_nadam_conv2d_2_bias_v/
+assignvariableop_52_nadam_conv2d_3_kernel_v-
)assignvariableop_53_nadam_conv2d_3_bias_v.
*assignvariableop_54_nadam_dense_4_kernel_v,
(assignvariableop_55_nadam_dense_4_bias_v.
*assignvariableop_56_nadam_dense_5_kernel_v,
(assignvariableop_57_nadam_dense_5_bias_v.
*assignvariableop_58_nadam_dense_6_kernel_v,
(assignvariableop_59_nadam_dense_6_bias_v.
*assignvariableop_60_nadam_dense_7_kernel_v,
(assignvariableop_61_nadam_dense_7_bias_v.
*assignvariableop_62_nadam_dense_8_kernel_v,
(assignvariableop_63_nadam_dense_8_bias_v.
*assignvariableop_64_nadam_dense_9_kernel_v,
(assignvariableop_65_nadam_dense_9_bias_v
identity_67??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?#
value?#B?#CB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?
value?B?CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_6_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_7_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_8_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_8_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_9_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_9_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_nadam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_nadam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_nadam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_nadam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_nadam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_nadam_momentum_cacheIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp"assignvariableop_26_true_positivesIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp"assignvariableop_27_true_negativesIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp#assignvariableop_28_false_positivesIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp#assignvariableop_29_false_negativesIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp$assignvariableop_30_true_positives_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp%assignvariableop_31_false_positives_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp$assignvariableop_32_true_positives_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_false_negatives_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_nadam_conv2d_2_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_nadam_conv2d_2_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_nadam_conv2d_3_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_nadam_conv2d_3_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_nadam_dense_4_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_nadam_dense_4_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_nadam_dense_5_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_nadam_dense_5_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_nadam_dense_6_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_nadam_dense_6_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_nadam_dense_7_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_nadam_dense_7_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_nadam_dense_8_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_nadam_dense_8_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_nadam_dense_9_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_nadam_dense_9_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp+assignvariableop_50_nadam_conv2d_2_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp)assignvariableop_51_nadam_conv2d_2_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp+assignvariableop_52_nadam_conv2d_3_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp)assignvariableop_53_nadam_conv2d_3_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp*assignvariableop_54_nadam_dense_4_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp(assignvariableop_55_nadam_dense_4_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp*assignvariableop_56_nadam_dense_5_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp(assignvariableop_57_nadam_dense_5_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_nadam_dense_6_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp(assignvariableop_59_nadam_dense_6_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_nadam_dense_7_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp(assignvariableop_61_nadam_dense_7_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_nadam_dense_8_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp(assignvariableop_63_nadam_dense_8_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp*assignvariableop_64_nadam_dense_9_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp(assignvariableop_65_nadam_dense_9_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_659
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_66Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_66?
Identity_67IdentityIdentity_66:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_67"#
identity_67Identity_67:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

*__inference_dense_8_layer_call_fn_11622891

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
GPU2*3J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_116219182
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
?
l
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_11622856

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_uniform/max?
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2
random_uniform/RandomUniform?
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub?
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
random_uniform/mul?
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
random_uniforme
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
GreaterEqual/y?
GreaterEqualGreaterEqualrandom_uniform:z:0GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
CastV
mulMulinputsCast:y:0*
T0*(
_output_shapes
:??????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x^
subSubsub/x:output:0Cast:y:0*
T0*(
_output_shapes
:??????????2
subW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2	
mul_1/xc
mul_1Mulmul_1/x:output:0sub:z:0*
T0*(
_output_shapes
:??????????2
mul_1Z
addAddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
addW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2	
mul_2/xc
mul_2Mulmul_2/x:output:0add:z:0*
T0*(
_output_shapes
:??????????2
mul_2W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2	
add_1/yg
add_1AddV2	mul_2:z:0add_1/y:output:0*
T0*(
_output_shapes
:??????????2
add_1^
IdentityIdentity	add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?@
?
C__inference_model_layer_call_and_return_conditional_losses_11622197

inputs
conv2d_2_11622149
conv2d_2_11622151
conv2d_3_11622154
conv2d_3_11622156
dense_4_11622161
dense_4_11622163
dense_5_11622167
dense_5_11622169
dense_6_11622173
dense_6_11622175
dense_7_11622179
dense_7_11622181
dense_8_11622185
dense_8_11622187
dense_9_11622191
dense_9_11622193
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_11622149conv2d_2_11622151*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_116215732"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_11622154conv2d_3_11622156*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_116216002"
 conv2d_3/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_116215522!
max_pooling2d_1/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_116216232
flatten_1/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_11622161dense_4_11622163*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_116216422!
dense_4/StatefulPartitionedCall?
alpha_dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_116216872!
alpha_dropout_4/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(alpha_dropout_4/PartitionedCall:output:0dense_5_11622167dense_5_11622169*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_116217112!
dense_5/StatefulPartitionedCall?
alpha_dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_116217562!
alpha_dropout_5/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall(alpha_dropout_5/PartitionedCall:output:0dense_6_11622173dense_6_11622175*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_116217802!
dense_6/StatefulPartitionedCall?
alpha_dropout_6/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_116218252!
alpha_dropout_6/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(alpha_dropout_6/PartitionedCall:output:0dense_7_11622179dense_7_11622181*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_116218492!
dense_7/StatefulPartitionedCall?
alpha_dropout_7/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_116218942!
alpha_dropout_7/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall(alpha_dropout_7/PartitionedCall:output:0dense_8_11622185dense_8_11622187*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_116219182!
dense_8/StatefulPartitionedCall?
alpha_dropout_8/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_116219632!
alpha_dropout_8/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(alpha_dropout_8/PartitionedCall:output:0dense_9_11622191dense_9_11622193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_116219872!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????(2
 
_user_specified_nameinputs
?
i
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_11622743

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
!__inference__traced_save_11623171
file_prefix.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop0
,savev2_false_positives_1_read_readvariableop/
+savev2_true_positives_2_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop6
2savev2_nadam_conv2d_2_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_2_bias_m_read_readvariableop6
2savev2_nadam_conv2d_3_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_3_bias_m_read_readvariableop5
1savev2_nadam_dense_4_kernel_m_read_readvariableop3
/savev2_nadam_dense_4_bias_m_read_readvariableop5
1savev2_nadam_dense_5_kernel_m_read_readvariableop3
/savev2_nadam_dense_5_bias_m_read_readvariableop5
1savev2_nadam_dense_6_kernel_m_read_readvariableop3
/savev2_nadam_dense_6_bias_m_read_readvariableop5
1savev2_nadam_dense_7_kernel_m_read_readvariableop3
/savev2_nadam_dense_7_bias_m_read_readvariableop5
1savev2_nadam_dense_8_kernel_m_read_readvariableop3
/savev2_nadam_dense_8_bias_m_read_readvariableop5
1savev2_nadam_dense_9_kernel_m_read_readvariableop3
/savev2_nadam_dense_9_bias_m_read_readvariableop6
2savev2_nadam_conv2d_2_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_2_bias_v_read_readvariableop6
2savev2_nadam_conv2d_3_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_3_bias_v_read_readvariableop5
1savev2_nadam_dense_4_kernel_v_read_readvariableop3
/savev2_nadam_dense_4_bias_v_read_readvariableop5
1savev2_nadam_dense_5_kernel_v_read_readvariableop3
/savev2_nadam_dense_5_bias_v_read_readvariableop5
1savev2_nadam_dense_6_kernel_v_read_readvariableop3
/savev2_nadam_dense_6_bias_v_read_readvariableop5
1savev2_nadam_dense_7_kernel_v_read_readvariableop3
/savev2_nadam_dense_7_bias_v_read_readvariableop5
1savev2_nadam_dense_8_kernel_v_read_readvariableop3
/savev2_nadam_dense_8_bias_v_read_readvariableop5
1savev2_nadam_dense_9_kernel_v_read_readvariableop3
/savev2_nadam_dense_9_bias_v_read_readvariableop
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
ShardedFilename?$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?#
value?#B?#CB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?
value?B?CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop+savev2_true_positives_2_read_readvariableop,savev2_false_negatives_1_read_readvariableop2savev2_nadam_conv2d_2_kernel_m_read_readvariableop0savev2_nadam_conv2d_2_bias_m_read_readvariableop2savev2_nadam_conv2d_3_kernel_m_read_readvariableop0savev2_nadam_conv2d_3_bias_m_read_readvariableop1savev2_nadam_dense_4_kernel_m_read_readvariableop/savev2_nadam_dense_4_bias_m_read_readvariableop1savev2_nadam_dense_5_kernel_m_read_readvariableop/savev2_nadam_dense_5_bias_m_read_readvariableop1savev2_nadam_dense_6_kernel_m_read_readvariableop/savev2_nadam_dense_6_bias_m_read_readvariableop1savev2_nadam_dense_7_kernel_m_read_readvariableop/savev2_nadam_dense_7_bias_m_read_readvariableop1savev2_nadam_dense_8_kernel_m_read_readvariableop/savev2_nadam_dense_8_bias_m_read_readvariableop1savev2_nadam_dense_9_kernel_m_read_readvariableop/savev2_nadam_dense_9_bias_m_read_readvariableop2savev2_nadam_conv2d_2_kernel_v_read_readvariableop0savev2_nadam_conv2d_2_bias_v_read_readvariableop2savev2_nadam_conv2d_3_kernel_v_read_readvariableop0savev2_nadam_conv2d_3_bias_v_read_readvariableop1savev2_nadam_dense_4_kernel_v_read_readvariableop/savev2_nadam_dense_4_bias_v_read_readvariableop1savev2_nadam_dense_5_kernel_v_read_readvariableop/savev2_nadam_dense_5_bias_v_read_readvariableop1savev2_nadam_dense_6_kernel_v_read_readvariableop/savev2_nadam_dense_6_bias_v_read_readvariableop1savev2_nadam_dense_7_kernel_v_read_readvariableop/savev2_nadam_dense_7_bias_v_read_readvariableop1savev2_nadam_dense_8_kernel_v_read_readvariableop/savev2_nadam_dense_8_bias_v_read_readvariableop1savev2_nadam_dense_9_kernel_v_read_readvariableop/savev2_nadam_dense_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Q
dtypesG
E2C	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :  : :
?}?:?:
??:?:
??:?:
??:?:
??:?:	?	:	: : : : : : : : : : :?:?:?:?::::: : :  : :
?}?:?:
??:?:
??:?:
??:?:
??:?:	?	:	: : :  : :
?}?:?:
??:?:
??:?:
??:?:
??:?:	?	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :&"
 
_output_shapes
:
?}?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?	: 

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
::,#(
&
_output_shapes
: : $

_output_shapes
: :,%(
&
_output_shapes
:  : &

_output_shapes
: :&'"
 
_output_shapes
:
?}?:!(

_output_shapes	
:?:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:&/"
 
_output_shapes
:
??:!0

_output_shapes	
:?:%1!

_output_shapes
:	?	: 2

_output_shapes
:	:,3(
&
_output_shapes
: : 4

_output_shapes
: :,5(
&
_output_shapes
:  : 6

_output_shapes
: :&7"
 
_output_shapes
:
?}?:!8

_output_shapes	
:?:&9"
 
_output_shapes
:
??:!:

_output_shapes	
:?:&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:&="
 
_output_shapes
:
??:!>

_output_shapes	
:?:&?"
 
_output_shapes
:
??:!@

_output_shapes	
:?:%A!

_output_shapes
:	?	: B

_output_shapes
:	:C

_output_shapes
: 
?
N
2__inference_alpha_dropout_4_layer_call_fn_11622694

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_116216872
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
(__inference_model_layer_call_fn_11622584

inputs
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

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*3J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_116221972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(2
 
_user_specified_nameinputs
?
N
2__inference_alpha_dropout_7_layer_call_fn_11622871

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_116218942
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
N
2__inference_alpha_dropout_6_layer_call_fn_11622812

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_116218252
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
&__inference_signature_wrapper_11622279
input_2
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

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*3J 8? *,
f'R%
#__inference__wrapped_model_116215462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(2
!
_user_specified_name	input_2
?

?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_11622595

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(2 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(2
 
_user_specified_nameinputs
?
i
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_11622802

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?S
?

C__inference_model_layer_call_and_return_conditional_losses_11622510

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2 2
conv2d_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2 2
conv2d_3/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPools
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????>  2
flatten_1/Const?
flatten_1/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????}2
flatten_1/Reshape?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?}?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
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
alpha_dropout_4/ShapeShapedense_4/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_4/Shape?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Selu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
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
dense_5/Selux
alpha_dropout_5/ShapeShapedense_5/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_5/Shape?
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
dense_6/Selux
alpha_dropout_6/ShapeShapedense_6/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_6/Shape?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Selu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAddq
dense_7/SeluSeludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_7/Selux
alpha_dropout_7/ShapeShapedense_7/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_7/Shape?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldense_7/Selu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/BiasAddq
dense_8/SeluSeludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_8/Selux
alpha_dropout_8/ShapeShapedense_8/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_8/Shape?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMuldense_8/Selu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_9/BiasAddy
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
dense_9/Softmax?
IdentityIdentitydense_9/Softmax:softmax:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????(2
 
_user_specified_nameinputs
?
i
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_11621687

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
k
2__inference_alpha_dropout_6_layer_call_fn_11622807

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_116218202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_11622861

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dense_7_layer_call_fn_11622832

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
GPU2*3J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_116218492
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
?
l
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_11621958

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_uniform/max?
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?Ѡ2
random_uniform/RandomUniform?
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub?
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
random_uniform/mul?
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
random_uniforme
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
GreaterEqual/y?
GreaterEqualGreaterEqualrandom_uniform:z:0GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
CastV
mulMulinputsCast:y:0*
T0*(
_output_shapes
:??????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x^
subSubsub/x:output:0Cast:y:0*
T0*(
_output_shapes
:??????????2
subW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2	
mul_1/xc
mul_1Mulmul_1/x:output:0sub:z:0*
T0*(
_output_shapes
:??????????2
mul_1Z
addAddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
addW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2	
mul_2/xc
mul_2Mulmul_2/x:output:0add:z:0*
T0*(
_output_shapes
:??????????2
mul_2W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2	
add_1/yg
add_1AddV2	mul_2:z:0add_1/y:output:0*
T0*(
_output_shapes
:??????????2
add_1^
IdentityIdentity	add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_11621756

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_8_layer_call_and_return_conditional_losses_11621918

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
?
l
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_11621682

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_uniform/max?
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2
random_uniform/RandomUniform?
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub?
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
random_uniform/mul?
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
random_uniforme
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
GreaterEqual/y?
GreaterEqualGreaterEqualrandom_uniform:z:0GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
CastV
mulMulinputsCast:y:0*
T0*(
_output_shapes
:??????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x^
subSubsub/x:output:0Cast:y:0*
T0*(
_output_shapes
:??????????2
subW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2	
mul_1/xc
mul_1Mulmul_1/x:output:0sub:z:0*
T0*(
_output_shapes
:??????????2
mul_1Z
addAddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
addW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2	
mul_2/xc
mul_2Mulmul_2/x:output:0add:z:0*
T0*(
_output_shapes
:??????????2
mul_2W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2	
add_1/yg
add_1AddV2	mul_2:z:0add_1/y:output:0*
T0*(
_output_shapes
:??????????2
add_1^
IdentityIdentity	add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_5_layer_call_and_return_conditional_losses_11621711

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
?
N
2__inference_alpha_dropout_5_layer_call_fn_11622753

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_116217562
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dense_5_layer_call_fn_11622714

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
GPU2*3J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_116217112
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
?
k
2__inference_alpha_dropout_5_layer_call_fn_11622748

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_116217512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_6_layer_call_and_return_conditional_losses_11621780

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
?
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_11622630

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????>  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????}2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????}2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_9_layer_call_and_return_conditional_losses_11621987

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

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
E__inference_dense_7_layer_call_and_return_conditional_losses_11622823

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
?
N
2__inference_max_pooling2d_1_layer_call_fn_11621558

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_116215522
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_6_layer_call_and_return_conditional_losses_11622764

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
?
N
2__inference_alpha_dropout_8_layer_call_fn_11622930

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_116219632
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?I
?
C__inference_model_layer_call_and_return_conditional_losses_11622004
input_2
conv2d_2_11621584
conv2d_2_11621586
conv2d_3_11621611
conv2d_3_11621613
dense_4_11621653
dense_4_11621655
dense_5_11621722
dense_5_11621724
dense_6_11621791
dense_6_11621793
dense_7_11621860
dense_7_11621862
dense_8_11621929
dense_8_11621931
dense_9_11621998
dense_9_11622000
identity??'alpha_dropout_4/StatefulPartitionedCall?'alpha_dropout_5/StatefulPartitionedCall?'alpha_dropout_6/StatefulPartitionedCall?'alpha_dropout_7/StatefulPartitionedCall?'alpha_dropout_8/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_2_11621584conv2d_2_11621586*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_116215732"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_11621611conv2d_3_11621613*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_116216002"
 conv2d_3/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_116215522!
max_pooling2d_1/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????}* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_116216232
flatten_1/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_11621653dense_4_11621655*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_116216422!
dense_4/StatefulPartitionedCall?
'alpha_dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_116216822)
'alpha_dropout_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall0alpha_dropout_4/StatefulPartitionedCall:output:0dense_5_11621722dense_5_11621724*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_116217112!
dense_5/StatefulPartitionedCall?
'alpha_dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0(^alpha_dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_116217512)
'alpha_dropout_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall0alpha_dropout_5/StatefulPartitionedCall:output:0dense_6_11621791dense_6_11621793*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_116217802!
dense_6/StatefulPartitionedCall?
'alpha_dropout_6/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0(^alpha_dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_116218202)
'alpha_dropout_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall0alpha_dropout_6/StatefulPartitionedCall:output:0dense_7_11621860dense_7_11621862*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_116218492!
dense_7/StatefulPartitionedCall?
'alpha_dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0(^alpha_dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_116218892)
'alpha_dropout_7/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall0alpha_dropout_7/StatefulPartitionedCall:output:0dense_8_11621929dense_8_11621931*
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
GPU2*3J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_116219182!
dense_8/StatefulPartitionedCall?
'alpha_dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0(^alpha_dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_116219582)
'alpha_dropout_8/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall0alpha_dropout_8/StatefulPartitionedCall:output:0dense_9_11621998dense_9_11622000*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_116219872!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0(^alpha_dropout_4/StatefulPartitionedCall(^alpha_dropout_5/StatefulPartitionedCall(^alpha_dropout_6/StatefulPartitionedCall(^alpha_dropout_7/StatefulPartitionedCall(^alpha_dropout_8/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::2R
'alpha_dropout_4/StatefulPartitionedCall'alpha_dropout_4/StatefulPartitionedCall2R
'alpha_dropout_5/StatefulPartitionedCall'alpha_dropout_5/StatefulPartitionedCall2R
'alpha_dropout_6/StatefulPartitionedCall'alpha_dropout_6/StatefulPartitionedCall2R
'alpha_dropout_7/StatefulPartitionedCall'alpha_dropout_7/StatefulPartitionedCall2R
'alpha_dropout_8/StatefulPartitionedCall'alpha_dropout_8/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:X T
/
_output_shapes
:?????????(2
!
_user_specified_name	input_2
?
i
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_11622684

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
(__inference_model_layer_call_fn_11622232
input_2
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

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*3J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_116221972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(2
!
_user_specified_name	input_2
??
?

C__inference_model_layer_call_and_return_conditional_losses_11622442

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2 2
conv2d_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2 2
conv2d_3/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPools
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????>  2
flatten_1/Const?
flatten_1/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????}2
flatten_1/Reshape?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?}?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
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
alpha_dropout_4/ShapeShapedense_4/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_4/Shape?
"alpha_dropout_4/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"alpha_dropout_4/random_uniform/min?
"alpha_dropout_4/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"alpha_dropout_4/random_uniform/max?
,alpha_dropout_4/random_uniform/RandomUniformRandomUniformalpha_dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2.
,alpha_dropout_4/random_uniform/RandomUniform?
"alpha_dropout_4/random_uniform/subSub+alpha_dropout_4/random_uniform/max:output:0+alpha_dropout_4/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"alpha_dropout_4/random_uniform/sub?
"alpha_dropout_4/random_uniform/mulMul5alpha_dropout_4/random_uniform/RandomUniform:output:0&alpha_dropout_4/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2$
"alpha_dropout_4/random_uniform/mul?
alpha_dropout_4/random_uniformAdd&alpha_dropout_4/random_uniform/mul:z:0+alpha_dropout_4/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2 
alpha_dropout_4/random_uniform?
alpha_dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
alpha_dropout_4/GreaterEqual/y?
alpha_dropout_4/GreaterEqualGreaterEqual"alpha_dropout_4/random_uniform:z:0'alpha_dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_4/GreaterEqual?
alpha_dropout_4/CastCast alpha_dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
alpha_dropout_4/Cast?
alpha_dropout_4/mulMuldense_4/Selu:activations:0alpha_dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_4/muls
alpha_dropout_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
alpha_dropout_4/sub/x?
alpha_dropout_4/subSubalpha_dropout_4/sub/x:output:0alpha_dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_4/subw
alpha_dropout_4/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2
alpha_dropout_4/mul_1/x?
alpha_dropout_4/mul_1Mul alpha_dropout_4/mul_1/x:output:0alpha_dropout_4/sub:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_4/mul_1?
alpha_dropout_4/addAddV2alpha_dropout_4/mul:z:0alpha_dropout_4/mul_1:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_4/addw
alpha_dropout_4/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2
alpha_dropout_4/mul_2/x?
alpha_dropout_4/mul_2Mul alpha_dropout_4/mul_2/x:output:0alpha_dropout_4/add:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_4/mul_2w
alpha_dropout_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2
alpha_dropout_4/add_1/y?
alpha_dropout_4/add_1AddV2alpha_dropout_4/mul_2:z:0 alpha_dropout_4/add_1/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_4/add_1?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulalpha_dropout_4/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
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
dense_5/Selux
alpha_dropout_5/ShapeShapedense_5/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_5/Shape?
"alpha_dropout_5/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"alpha_dropout_5/random_uniform/min?
"alpha_dropout_5/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"alpha_dropout_5/random_uniform/max?
,alpha_dropout_5/random_uniform/RandomUniformRandomUniformalpha_dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??2.
,alpha_dropout_5/random_uniform/RandomUniform?
"alpha_dropout_5/random_uniform/subSub+alpha_dropout_5/random_uniform/max:output:0+alpha_dropout_5/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"alpha_dropout_5/random_uniform/sub?
"alpha_dropout_5/random_uniform/mulMul5alpha_dropout_5/random_uniform/RandomUniform:output:0&alpha_dropout_5/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2$
"alpha_dropout_5/random_uniform/mul?
alpha_dropout_5/random_uniformAdd&alpha_dropout_5/random_uniform/mul:z:0+alpha_dropout_5/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2 
alpha_dropout_5/random_uniform?
alpha_dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
alpha_dropout_5/GreaterEqual/y?
alpha_dropout_5/GreaterEqualGreaterEqual"alpha_dropout_5/random_uniform:z:0'alpha_dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_5/GreaterEqual?
alpha_dropout_5/CastCast alpha_dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
alpha_dropout_5/Cast?
alpha_dropout_5/mulMuldense_5/Selu:activations:0alpha_dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_5/muls
alpha_dropout_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
alpha_dropout_5/sub/x?
alpha_dropout_5/subSubalpha_dropout_5/sub/x:output:0alpha_dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_5/subw
alpha_dropout_5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2
alpha_dropout_5/mul_1/x?
alpha_dropout_5/mul_1Mul alpha_dropout_5/mul_1/x:output:0alpha_dropout_5/sub:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_5/mul_1?
alpha_dropout_5/addAddV2alpha_dropout_5/mul:z:0alpha_dropout_5/mul_1:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_5/addw
alpha_dropout_5/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2
alpha_dropout_5/mul_2/x?
alpha_dropout_5/mul_2Mul alpha_dropout_5/mul_2/x:output:0alpha_dropout_5/add:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_5/mul_2w
alpha_dropout_5/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2
alpha_dropout_5/add_1/y?
alpha_dropout_5/add_1AddV2alpha_dropout_5/mul_2:z:0 alpha_dropout_5/add_1/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_5/add_1?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulalpha_dropout_5/add_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
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
dense_6/Selux
alpha_dropout_6/ShapeShapedense_6/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_6/Shape?
"alpha_dropout_6/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"alpha_dropout_6/random_uniform/min?
"alpha_dropout_6/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"alpha_dropout_6/random_uniform/max?
,alpha_dropout_6/random_uniform/RandomUniformRandomUniformalpha_dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?Ğ2.
,alpha_dropout_6/random_uniform/RandomUniform?
"alpha_dropout_6/random_uniform/subSub+alpha_dropout_6/random_uniform/max:output:0+alpha_dropout_6/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"alpha_dropout_6/random_uniform/sub?
"alpha_dropout_6/random_uniform/mulMul5alpha_dropout_6/random_uniform/RandomUniform:output:0&alpha_dropout_6/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2$
"alpha_dropout_6/random_uniform/mul?
alpha_dropout_6/random_uniformAdd&alpha_dropout_6/random_uniform/mul:z:0+alpha_dropout_6/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2 
alpha_dropout_6/random_uniform?
alpha_dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
alpha_dropout_6/GreaterEqual/y?
alpha_dropout_6/GreaterEqualGreaterEqual"alpha_dropout_6/random_uniform:z:0'alpha_dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_6/GreaterEqual?
alpha_dropout_6/CastCast alpha_dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
alpha_dropout_6/Cast?
alpha_dropout_6/mulMuldense_6/Selu:activations:0alpha_dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_6/muls
alpha_dropout_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
alpha_dropout_6/sub/x?
alpha_dropout_6/subSubalpha_dropout_6/sub/x:output:0alpha_dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_6/subw
alpha_dropout_6/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2
alpha_dropout_6/mul_1/x?
alpha_dropout_6/mul_1Mul alpha_dropout_6/mul_1/x:output:0alpha_dropout_6/sub:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_6/mul_1?
alpha_dropout_6/addAddV2alpha_dropout_6/mul:z:0alpha_dropout_6/mul_1:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_6/addw
alpha_dropout_6/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2
alpha_dropout_6/mul_2/x?
alpha_dropout_6/mul_2Mul alpha_dropout_6/mul_2/x:output:0alpha_dropout_6/add:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_6/mul_2w
alpha_dropout_6/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2
alpha_dropout_6/add_1/y?
alpha_dropout_6/add_1AddV2alpha_dropout_6/mul_2:z:0 alpha_dropout_6/add_1/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_6/add_1?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulalpha_dropout_6/add_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAddq
dense_7/SeluSeludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_7/Selux
alpha_dropout_7/ShapeShapedense_7/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_7/Shape?
"alpha_dropout_7/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"alpha_dropout_7/random_uniform/min?
"alpha_dropout_7/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"alpha_dropout_7/random_uniform/max?
,alpha_dropout_7/random_uniform/RandomUniformRandomUniformalpha_dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2.
,alpha_dropout_7/random_uniform/RandomUniform?
"alpha_dropout_7/random_uniform/subSub+alpha_dropout_7/random_uniform/max:output:0+alpha_dropout_7/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"alpha_dropout_7/random_uniform/sub?
"alpha_dropout_7/random_uniform/mulMul5alpha_dropout_7/random_uniform/RandomUniform:output:0&alpha_dropout_7/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2$
"alpha_dropout_7/random_uniform/mul?
alpha_dropout_7/random_uniformAdd&alpha_dropout_7/random_uniform/mul:z:0+alpha_dropout_7/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2 
alpha_dropout_7/random_uniform?
alpha_dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
alpha_dropout_7/GreaterEqual/y?
alpha_dropout_7/GreaterEqualGreaterEqual"alpha_dropout_7/random_uniform:z:0'alpha_dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_7/GreaterEqual?
alpha_dropout_7/CastCast alpha_dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
alpha_dropout_7/Cast?
alpha_dropout_7/mulMuldense_7/Selu:activations:0alpha_dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_7/muls
alpha_dropout_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
alpha_dropout_7/sub/x?
alpha_dropout_7/subSubalpha_dropout_7/sub/x:output:0alpha_dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_7/subw
alpha_dropout_7/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2
alpha_dropout_7/mul_1/x?
alpha_dropout_7/mul_1Mul alpha_dropout_7/mul_1/x:output:0alpha_dropout_7/sub:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_7/mul_1?
alpha_dropout_7/addAddV2alpha_dropout_7/mul:z:0alpha_dropout_7/mul_1:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_7/addw
alpha_dropout_7/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2
alpha_dropout_7/mul_2/x?
alpha_dropout_7/mul_2Mul alpha_dropout_7/mul_2/x:output:0alpha_dropout_7/add:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_7/mul_2w
alpha_dropout_7/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2
alpha_dropout_7/add_1/y?
alpha_dropout_7/add_1AddV2alpha_dropout_7/mul_2:z:0 alpha_dropout_7/add_1/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_7/add_1?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulalpha_dropout_7/add_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/BiasAddq
dense_8/SeluSeludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_8/Selux
alpha_dropout_8/ShapeShapedense_8/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_8/Shape?
"alpha_dropout_8/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"alpha_dropout_8/random_uniform/min?
"alpha_dropout_8/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"alpha_dropout_8/random_uniform/max?
,alpha_dropout_8/random_uniform/RandomUniformRandomUniformalpha_dropout_8/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2.
,alpha_dropout_8/random_uniform/RandomUniform?
"alpha_dropout_8/random_uniform/subSub+alpha_dropout_8/random_uniform/max:output:0+alpha_dropout_8/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"alpha_dropout_8/random_uniform/sub?
"alpha_dropout_8/random_uniform/mulMul5alpha_dropout_8/random_uniform/RandomUniform:output:0&alpha_dropout_8/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2$
"alpha_dropout_8/random_uniform/mul?
alpha_dropout_8/random_uniformAdd&alpha_dropout_8/random_uniform/mul:z:0+alpha_dropout_8/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2 
alpha_dropout_8/random_uniform?
alpha_dropout_8/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
alpha_dropout_8/GreaterEqual/y?
alpha_dropout_8/GreaterEqualGreaterEqual"alpha_dropout_8/random_uniform:z:0'alpha_dropout_8/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_8/GreaterEqual?
alpha_dropout_8/CastCast alpha_dropout_8/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
alpha_dropout_8/Cast?
alpha_dropout_8/mulMuldense_8/Selu:activations:0alpha_dropout_8/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_8/muls
alpha_dropout_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
alpha_dropout_8/sub/x?
alpha_dropout_8/subSubalpha_dropout_8/sub/x:output:0alpha_dropout_8/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_8/subw
alpha_dropout_8/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2
alpha_dropout_8/mul_1/x?
alpha_dropout_8/mul_1Mul alpha_dropout_8/mul_1/x:output:0alpha_dropout_8/sub:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_8/mul_1?
alpha_dropout_8/addAddV2alpha_dropout_8/mul:z:0alpha_dropout_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_8/addw
alpha_dropout_8/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2
alpha_dropout_8/mul_2/x?
alpha_dropout_8/mul_2Mul alpha_dropout_8/mul_2/x:output:0alpha_dropout_8/add:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_8/mul_2w
alpha_dropout_8/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2
alpha_dropout_8/add_1/y?
alpha_dropout_8/add_1AddV2alpha_dropout_8/mul_2:z:0 alpha_dropout_8/add_1/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_8/add_1?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulalpha_dropout_8/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_9/BiasAddy
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
dense_9/Softmax?
IdentityIdentitydense_9/Softmax:softmax:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????(2
 
_user_specified_nameinputs
?_
?
#__inference__wrapped_model_11621546
input_21
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource1
-model_conv2d_3_conv2d_readvariableop_resource2
.model_conv2d_3_biasadd_readvariableop_resource0
,model_dense_4_matmul_readvariableop_resource1
-model_dense_4_biasadd_readvariableop_resource0
,model_dense_5_matmul_readvariableop_resource1
-model_dense_5_biasadd_readvariableop_resource0
,model_dense_6_matmul_readvariableop_resource1
-model_dense_6_biasadd_readvariableop_resource0
,model_dense_7_matmul_readvariableop_resource1
-model_dense_7_biasadd_readvariableop_resource0
,model_dense_8_matmul_readvariableop_resource1
-model_dense_8_biasadd_readvariableop_resource0
,model_dense_9_matmul_readvariableop_resource1
-model_dense_9_biasadd_readvariableop_resource
identity??%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?%model/conv2d_3/BiasAdd/ReadVariableOp?$model/conv2d_3/Conv2D/ReadVariableOp?$model/dense_4/BiasAdd/ReadVariableOp?#model/dense_4/MatMul/ReadVariableOp?$model/dense_5/BiasAdd/ReadVariableOp?#model/dense_5/MatMul/ReadVariableOp?$model/dense_6/BiasAdd/ReadVariableOp?#model/dense_6/MatMul/ReadVariableOp?$model/dense_7/BiasAdd/ReadVariableOp?#model/dense_7/MatMul/ReadVariableOp?$model/dense_8/BiasAdd/ReadVariableOp?#model/dense_8/MatMul/ReadVariableOp?$model/dense_9/BiasAdd/ReadVariableOp?#model/dense_9/MatMul/ReadVariableOp?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp?
model/conv2d_2/Conv2DConv2Dinput_2,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 *
paddingSAME*
strides
2
model/conv2d_2/Conv2D?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 2
model/conv2d_2/BiasAdd?
model/conv2d_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2 2
model/conv2d_2/Relu?
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02&
$model/conv2d_3/Conv2D/ReadVariableOp?
model/conv2d_3/Conv2DConv2D!model/conv2d_2/Relu:activations:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 *
paddingSAME*
strides
2
model/conv2d_3/Conv2D?
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d_3/BiasAdd/ReadVariableOp?
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 2
model/conv2d_3/BiasAdd?
model/conv2d_3/ReluRelumodel/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2 2
model/conv2d_3/Relu?
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_3/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????>  2
model/flatten_1/Const?
model/flatten_1/ReshapeReshape&model/max_pooling2d_1/MaxPool:output:0model/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????}2
model/flatten_1/Reshape?
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?}?*
dtype02%
#model/dense_4/MatMul/ReadVariableOp?
model/dense_4/MatMulMatMul model/flatten_1/Reshape:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_4/MatMul?
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/dense_4/BiasAdd/ReadVariableOp?
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_4/BiasAdd?
model/dense_4/SeluSelumodel/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense_4/Selu?
model/alpha_dropout_4/ShapeShape model/dense_4/Selu:activations:0*
T0*
_output_shapes
:2
model/alpha_dropout_4/Shape?
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#model/dense_5/MatMul/ReadVariableOp?
model/dense_5/MatMulMatMul model/dense_4/Selu:activations:0+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_5/MatMul?
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/dense_5/BiasAdd/ReadVariableOp?
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_5/BiasAdd?
model/dense_5/SeluSelumodel/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense_5/Selu?
model/alpha_dropout_5/ShapeShape model/dense_5/Selu:activations:0*
T0*
_output_shapes
:2
model/alpha_dropout_5/Shape?
#model/dense_6/MatMul/ReadVariableOpReadVariableOp,model_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#model/dense_6/MatMul/ReadVariableOp?
model/dense_6/MatMulMatMul model/dense_5/Selu:activations:0+model/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_6/MatMul?
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/dense_6/BiasAdd/ReadVariableOp?
model/dense_6/BiasAddBiasAddmodel/dense_6/MatMul:product:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_6/BiasAdd?
model/dense_6/SeluSelumodel/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense_6/Selu?
model/alpha_dropout_6/ShapeShape model/dense_6/Selu:activations:0*
T0*
_output_shapes
:2
model/alpha_dropout_6/Shape?
#model/dense_7/MatMul/ReadVariableOpReadVariableOp,model_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#model/dense_7/MatMul/ReadVariableOp?
model/dense_7/MatMulMatMul model/dense_6/Selu:activations:0+model/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_7/MatMul?
$model/dense_7/BiasAdd/ReadVariableOpReadVariableOp-model_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/dense_7/BiasAdd/ReadVariableOp?
model/dense_7/BiasAddBiasAddmodel/dense_7/MatMul:product:0,model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_7/BiasAdd?
model/dense_7/SeluSelumodel/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense_7/Selu?
model/alpha_dropout_7/ShapeShape model/dense_7/Selu:activations:0*
T0*
_output_shapes
:2
model/alpha_dropout_7/Shape?
#model/dense_8/MatMul/ReadVariableOpReadVariableOp,model_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#model/dense_8/MatMul/ReadVariableOp?
model/dense_8/MatMulMatMul model/dense_7/Selu:activations:0+model/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_8/MatMul?
$model/dense_8/BiasAdd/ReadVariableOpReadVariableOp-model_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/dense_8/BiasAdd/ReadVariableOp?
model/dense_8/BiasAddBiasAddmodel/dense_8/MatMul:product:0,model/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_8/BiasAdd?
model/dense_8/SeluSelumodel/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense_8/Selu?
model/alpha_dropout_8/ShapeShape model/dense_8/Selu:activations:0*
T0*
_output_shapes
:2
model/alpha_dropout_8/Shape?
#model/dense_9/MatMul/ReadVariableOpReadVariableOp,model_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02%
#model/dense_9/MatMul/ReadVariableOp?
model/dense_9/MatMulMatMul model/dense_8/Selu:activations:0+model/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
model/dense_9/MatMul?
$model/dense_9/BiasAdd/ReadVariableOpReadVariableOp-model_dense_9_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02&
$model/dense_9/BiasAdd/ReadVariableOp?
model/dense_9/BiasAddBiasAddmodel/dense_9/MatMul:product:0,model/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
model/dense_9/BiasAdd?
model/dense_9/SoftmaxSoftmaxmodel/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
model/dense_9/Softmax?
IdentityIdentitymodel/dense_9/Softmax:softmax:0&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp$^model/dense_5/MatMul/ReadVariableOp%^model/dense_6/BiasAdd/ReadVariableOp$^model/dense_6/MatMul/ReadVariableOp%^model/dense_7/BiasAdd/ReadVariableOp$^model/dense_7/MatMul/ReadVariableOp%^model/dense_8/BiasAdd/ReadVariableOp$^model/dense_8/MatMul/ReadVariableOp%^model/dense_9/BiasAdd/ReadVariableOp$^model/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2J
#model/dense_5/MatMul/ReadVariableOp#model/dense_5/MatMul/ReadVariableOp2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2J
#model/dense_6/MatMul/ReadVariableOp#model/dense_6/MatMul/ReadVariableOp2L
$model/dense_7/BiasAdd/ReadVariableOp$model/dense_7/BiasAdd/ReadVariableOp2J
#model/dense_7/MatMul/ReadVariableOp#model/dense_7/MatMul/ReadVariableOp2L
$model/dense_8/BiasAdd/ReadVariableOp$model/dense_8/BiasAdd/ReadVariableOp2J
#model/dense_8/MatMul/ReadVariableOp#model/dense_8/MatMul/ReadVariableOp2L
$model/dense_9/BiasAdd/ReadVariableOp$model/dense_9/BiasAdd/ReadVariableOp2J
#model/dense_9/MatMul/ReadVariableOp#model/dense_9/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????(2
!
_user_specified_name	input_2
?	
?
E__inference_dense_4_layer_call_and_return_conditional_losses_11622646

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?}?*
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
:??????????}::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????}
 
_user_specified_nameinputs
?

?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_11621600

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(2 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(2 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(2 
 
_user_specified_nameinputs
?
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_11621623

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????>  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????}2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????}2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
k
2__inference_alpha_dropout_4_layer_call_fn_11622689

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_116216822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_11622679

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_uniform/max?
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ϲ2
random_uniform/RandomUniform?
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub?
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
random_uniform/mul?
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
random_uniforme
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
GreaterEqual/y?
GreaterEqualGreaterEqualrandom_uniform:z:0GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
CastV
mulMulinputsCast:y:0*
T0*(
_output_shapes
:??????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x^
subSubsub/x:output:0Cast:y:0*
T0*(
_output_shapes
:??????????2
subW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2	
mul_1/xc
mul_1Mulmul_1/x:output:0sub:z:0*
T0*(
_output_shapes
:??????????2
mul_1Z
addAddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
addW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2	
mul_2/xc
mul_2Mulmul_2/x:output:0add:z:0*
T0*(
_output_shapes
:??????????2
mul_2W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2	
add_1/yg
add_1AddV2	mul_2:z:0add_1/y:output:0*
T0*(
_output_shapes
:??????????2
add_1^
IdentityIdentity	add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dense_6_layer_call_fn_11622773

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
GPU2*3J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_116217802
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
?

*__inference_dense_9_layer_call_fn_11622950

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
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_116219872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_11621573

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(2 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(2
 
_user_specified_nameinputs
?	
?
E__inference_dense_7_layer_call_and_return_conditional_losses_11621849

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
?
k
2__inference_alpha_dropout_8_layer_call_fn_11622925

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*3J 8? *V
fQRO
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_116219582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_11621889

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_uniform/max?
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2
random_uniform/RandomUniform?
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub?
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
random_uniform/mul?
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
random_uniforme
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
GreaterEqual/y?
GreaterEqualGreaterEqualrandom_uniform:z:0GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
CastV
mulMulinputsCast:y:0*
T0*(
_output_shapes
:??????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x^
subSubsub/x:output:0Cast:y:0*
T0*(
_output_shapes
:??????????2
subW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2	
mul_1/xc
mul_1Mulmul_1/x:output:0sub:z:0*
T0*(
_output_shapes
:??????????2
mul_1Z
addAddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
addW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2	
mul_2/xc
mul_2Mulmul_2/x:output:0add:z:0*
T0*(
_output_shapes
:??????????2
mul_2W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2	
add_1/yg
add_1AddV2	mul_2:z:0add_1/y:output:0*
T0*(
_output_shapes
:??????????2
add_1^
IdentityIdentity	add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_11621552

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_11622615

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(2 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(2 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(2 
 
_user_specified_nameinputs
?

*__inference_dense_4_layer_call_fn_11622655

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
GPU2*3J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_116216422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????}::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????}
 
_user_specified_nameinputs
?
l
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_11621820

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_uniform/max?
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2
random_uniform/RandomUniform?
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub?
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
random_uniform/mul?
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
random_uniforme
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
GreaterEqual/y?
GreaterEqualGreaterEqualrandom_uniform:z:0GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
CastV
mulMulinputsCast:y:0*
T0*(
_output_shapes
:??????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x^
subSubsub/x:output:0Cast:y:0*
T0*(
_output_shapes
:??????????2
subW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2	
mul_1/xc
mul_1Mulmul_1/x:output:0sub:z:0*
T0*(
_output_shapes
:??????????2
mul_1Z
addAddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
addW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2	
mul_2/xc
mul_2Mulmul_2/x:output:0add:z:0*
T0*(
_output_shapes
:??????????2
mul_2W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2	
add_1/yg
add_1AddV2	mul_2:z:0add_1/y:output:0*
T0*(
_output_shapes
:??????????2
add_1^
IdentityIdentity	add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_8_layer_call_and_return_conditional_losses_11622882

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

?
(__inference_model_layer_call_fn_11622144
input_2
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

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*3J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_116221092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(2::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(2
!
_user_specified_name	input_2
?
?
+__inference_conv2d_3_layer_call_fn_11622624

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
 */
_output_shapes
:?????????(2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*3J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_116216002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(2 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(2 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(2 
 
_user_specified_nameinputs
?
i
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_11621894

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
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
C
input_28
serving_default_input_2:0?????????(2;
dense_90
StatefulPartitionedCall:0?????????	tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
	optimizer

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_networkύ{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 50, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_4", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["alpha_dropout_4", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_5", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["alpha_dropout_5", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_6", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_6", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["alpha_dropout_6", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_7", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_7", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["alpha_dropout_7", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_8", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 9, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["alpha_dropout_8", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_9", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 40, 50, 2]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 50, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 50, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_4", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["alpha_dropout_4", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_5", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["alpha_dropout_5", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_6", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_6", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["alpha_dropout_6", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_7", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_7", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["alpha_dropout_7", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_8", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 9, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["alpha_dropout_8", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_9", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "CategoricalAccuracy", "config": {"name": "categorical_accuracy", "dtype": "float32"}}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}, {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 9.99999993922529e-09, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 50, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 50, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?


kernel
bias
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 50, 2]}}
?


 kernel
!bias
#"_self_saveable_object_factories
#trainable_variables
$	variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 50, 32]}}
?
#'_self_saveable_object_factories
(trainable_variables
)	variables
*regularization_losses
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
#,_self_saveable_object_factories
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

1kernel
2bias
#3_self_saveable_object_factories
4trainable_variables
5	variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16000]}}
?
#8_self_saveable_object_factories
9trainable_variables
:	variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AlphaDropout", "name": "alpha_dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "alpha_dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2}}
?

=kernel
>bias
#?_self_saveable_object_factories
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
#D_self_saveable_object_factories
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AlphaDropout", "name": "alpha_dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "alpha_dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2}}
?

Ikernel
Jbias
#K_self_saveable_object_factories
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
#P_self_saveable_object_factories
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AlphaDropout", "name": "alpha_dropout_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "alpha_dropout_6", "trainable": true, "dtype": "float32", "rate": 0.2}}
?

Ukernel
Vbias
#W_self_saveable_object_factories
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
#\_self_saveable_object_factories
]trainable_variables
^	variables
_regularization_losses
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AlphaDropout", "name": "alpha_dropout_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "alpha_dropout_7", "trainable": true, "dtype": "float32", "rate": 0.2}}
?

akernel
bbias
#c_self_saveable_object_factories
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
#h_self_saveable_object_factories
itrainable_variables
j	variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AlphaDropout", "name": "alpha_dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "alpha_dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2}}
?

mkernel
nbias
#o_self_saveable_object_factories
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 9, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
titer

ubeta_1

vbeta_2
	wdecay
xlearning_rate
ymomentum_cachem?m? m?!m?1m?2m?=m?>m?Im?Jm?Um?Vm?am?bm?mm?nm?v?v? v?!v?1v?2v?=v?>v?Iv?Jv?Uv?Vv?av?bv?mv?nv?"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
?
0
1
 2
!3
14
25
=6
>7
I8
J9
U10
V11
a12
b13
m14
n15"
trackable_list_wrapper
?
0
1
 2
!3
14
25
=6
>7
I8
J9
U10
V11
a12
b13
m14
n15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
zlayer_metrics
{non_trainable_variables
|metrics
	variables
}layer_regularization_losses

~layers
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
):' 2conv2d_2/kernel
: 2conv2d_2/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_metrics
?metrics
trainable_variables
	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_3/kernel
: 2conv2d_3/bias
 "
trackable_dict_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
#trainable_variables
$	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
%regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
(trainable_variables
)	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
*regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
-trainable_variables
.	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
?}?2dense_4/kernel
:?2dense_4/bias
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
4trainable_variables
5	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
6regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
9trainable_variables
:	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_5/kernel
:?2dense_5/bias
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
@trainable_variables
A	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Bregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
Etrainable_variables
F	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_6/kernel
:?2dense_6/bias
 "
trackable_dict_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
Ltrainable_variables
M	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
Qtrainable_variables
R	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_7/kernel
:?2dense_7/bias
 "
trackable_dict_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
Xtrainable_variables
Y	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Zregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
]trainable_variables
^	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_8/kernel
:?2dense_8/bias
 "
trackable_dict_wrapper
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
dtrainable_variables
e	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
itrainable_variables
j	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?	2dense_9/kernel
:	2dense_9/bias
 "
trackable_dict_wrapper
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
ptrainable_variables
q	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
H
?0
?1
?2
?3
?4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
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

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "CategoricalAccuracy", "name": "categorical_accuracy", "dtype": "float32", "config": {"name": "categorical_accuracy", "dtype": "float32"}}
?"
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api"?!
_tf_keras_metric?!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
?
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Nadam/conv2d_2/kernel/m
!: 2Nadam/conv2d_2/bias/m
/:-  2Nadam/conv2d_3/kernel/m
!: 2Nadam/conv2d_3/bias/m
(:&
?}?2Nadam/dense_4/kernel/m
!:?2Nadam/dense_4/bias/m
(:&
??2Nadam/dense_5/kernel/m
!:?2Nadam/dense_5/bias/m
(:&
??2Nadam/dense_6/kernel/m
!:?2Nadam/dense_6/bias/m
(:&
??2Nadam/dense_7/kernel/m
!:?2Nadam/dense_7/bias/m
(:&
??2Nadam/dense_8/kernel/m
!:?2Nadam/dense_8/bias/m
':%	?	2Nadam/dense_9/kernel/m
 :	2Nadam/dense_9/bias/m
/:- 2Nadam/conv2d_2/kernel/v
!: 2Nadam/conv2d_2/bias/v
/:-  2Nadam/conv2d_3/kernel/v
!: 2Nadam/conv2d_3/bias/v
(:&
?}?2Nadam/dense_4/kernel/v
!:?2Nadam/dense_4/bias/v
(:&
??2Nadam/dense_5/kernel/v
!:?2Nadam/dense_5/bias/v
(:&
??2Nadam/dense_6/kernel/v
!:?2Nadam/dense_6/bias/v
(:&
??2Nadam/dense_7/kernel/v
!:?2Nadam/dense_7/bias/v
(:&
??2Nadam/dense_8/kernel/v
!:?2Nadam/dense_8/bias/v
':%	?	2Nadam/dense_9/kernel/v
 :	2Nadam/dense_9/bias/v
?2?
(__inference_model_layer_call_fn_11622144
(__inference_model_layer_call_fn_11622547
(__inference_model_layer_call_fn_11622584
(__inference_model_layer_call_fn_11622232?
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
#__inference__wrapped_model_11621546?
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
annotations? *.?+
)?&
input_2?????????(2
?2?
C__inference_model_layer_call_and_return_conditional_losses_11622510
C__inference_model_layer_call_and_return_conditional_losses_11622004
C__inference_model_layer_call_and_return_conditional_losses_11622055
C__inference_model_layer_call_and_return_conditional_losses_11622442?
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
+__inference_conv2d_2_layer_call_fn_11622604?
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_11622595?
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
+__inference_conv2d_3_layer_call_fn_11622624?
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
F__inference_conv2d_3_layer_call_and_return_conditional_losses_11622615?
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
?2?
2__inference_max_pooling2d_1_layer_call_fn_11621558?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_11621552?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_flatten_1_layer_call_fn_11622635?
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_11622630?
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
*__inference_dense_4_layer_call_fn_11622655?
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
E__inference_dense_4_layer_call_and_return_conditional_losses_11622646?
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
?2?
2__inference_alpha_dropout_4_layer_call_fn_11622694
2__inference_alpha_dropout_4_layer_call_fn_11622689?
???
FullArgSpec)
args!?
jself
jinputs

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
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_11622679
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_11622684?
???
FullArgSpec)
args!?
jself
jinputs

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
?2?
*__inference_dense_5_layer_call_fn_11622714?
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
E__inference_dense_5_layer_call_and_return_conditional_losses_11622705?
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
?2?
2__inference_alpha_dropout_5_layer_call_fn_11622748
2__inference_alpha_dropout_5_layer_call_fn_11622753?
???
FullArgSpec)
args!?
jself
jinputs

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
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_11622738
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_11622743?
???
FullArgSpec)
args!?
jself
jinputs

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
?2?
*__inference_dense_6_layer_call_fn_11622773?
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
E__inference_dense_6_layer_call_and_return_conditional_losses_11622764?
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
?2?
2__inference_alpha_dropout_6_layer_call_fn_11622812
2__inference_alpha_dropout_6_layer_call_fn_11622807?
???
FullArgSpec)
args!?
jself
jinputs

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
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_11622797
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_11622802?
???
FullArgSpec)
args!?
jself
jinputs

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
?2?
*__inference_dense_7_layer_call_fn_11622832?
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
E__inference_dense_7_layer_call_and_return_conditional_losses_11622823?
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
?2?
2__inference_alpha_dropout_7_layer_call_fn_11622866
2__inference_alpha_dropout_7_layer_call_fn_11622871?
???
FullArgSpec)
args!?
jself
jinputs

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
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_11622861
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_11622856?
???
FullArgSpec)
args!?
jself
jinputs

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
?2?
*__inference_dense_8_layer_call_fn_11622891?
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
E__inference_dense_8_layer_call_and_return_conditional_losses_11622882?
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
?2?
2__inference_alpha_dropout_8_layer_call_fn_11622925
2__inference_alpha_dropout_8_layer_call_fn_11622930?
???
FullArgSpec)
args!?
jself
jinputs

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
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_11622920
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_11622915?
???
FullArgSpec)
args!?
jself
jinputs

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
?2?
*__inference_dense_9_layer_call_fn_11622950?
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
E__inference_dense_9_layer_call_and_return_conditional_losses_11622941?
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
&__inference_signature_wrapper_11622279input_2"?
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
 ?
#__inference__wrapped_model_11621546 !12=>IJUVabmn8?5
.?+
)?&
input_2?????????(2
? "1?.
,
dense_9!?
dense_9?????????	?
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_11622679^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
M__inference_alpha_dropout_4_layer_call_and_return_conditional_losses_11622684^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
2__inference_alpha_dropout_4_layer_call_fn_11622689Q4?1
*?'
!?
inputs??????????
p
? "????????????
2__inference_alpha_dropout_4_layer_call_fn_11622694Q4?1
*?'
!?
inputs??????????
p 
? "????????????
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_11622738^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
M__inference_alpha_dropout_5_layer_call_and_return_conditional_losses_11622743^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
2__inference_alpha_dropout_5_layer_call_fn_11622748Q4?1
*?'
!?
inputs??????????
p
? "????????????
2__inference_alpha_dropout_5_layer_call_fn_11622753Q4?1
*?'
!?
inputs??????????
p 
? "????????????
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_11622797^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
M__inference_alpha_dropout_6_layer_call_and_return_conditional_losses_11622802^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
2__inference_alpha_dropout_6_layer_call_fn_11622807Q4?1
*?'
!?
inputs??????????
p
? "????????????
2__inference_alpha_dropout_6_layer_call_fn_11622812Q4?1
*?'
!?
inputs??????????
p 
? "????????????
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_11622856^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
M__inference_alpha_dropout_7_layer_call_and_return_conditional_losses_11622861^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
2__inference_alpha_dropout_7_layer_call_fn_11622866Q4?1
*?'
!?
inputs??????????
p
? "????????????
2__inference_alpha_dropout_7_layer_call_fn_11622871Q4?1
*?'
!?
inputs??????????
p 
? "????????????
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_11622915^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
M__inference_alpha_dropout_8_layer_call_and_return_conditional_losses_11622920^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
2__inference_alpha_dropout_8_layer_call_fn_11622925Q4?1
*?'
!?
inputs??????????
p
? "????????????
2__inference_alpha_dropout_8_layer_call_fn_11622930Q4?1
*?'
!?
inputs??????????
p 
? "????????????
F__inference_conv2d_2_layer_call_and_return_conditional_losses_11622595l7?4
-?*
(?%
inputs?????????(2
? "-?*
#? 
0?????????(2 
? ?
+__inference_conv2d_2_layer_call_fn_11622604_7?4
-?*
(?%
inputs?????????(2
? " ??????????(2 ?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_11622615l !7?4
-?*
(?%
inputs?????????(2 
? "-?*
#? 
0?????????(2 
? ?
+__inference_conv2d_3_layer_call_fn_11622624_ !7?4
-?*
(?%
inputs?????????(2 
? " ??????????(2 ?
E__inference_dense_4_layer_call_and_return_conditional_losses_11622646^120?-
&?#
!?
inputs??????????}
? "&?#
?
0??????????
? 
*__inference_dense_4_layer_call_fn_11622655Q120?-
&?#
!?
inputs??????????}
? "????????????
E__inference_dense_5_layer_call_and_return_conditional_losses_11622705^=>0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_5_layer_call_fn_11622714Q=>0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_6_layer_call_and_return_conditional_losses_11622764^IJ0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_6_layer_call_fn_11622773QIJ0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_7_layer_call_and_return_conditional_losses_11622823^UV0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_7_layer_call_fn_11622832QUV0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_8_layer_call_and_return_conditional_losses_11622882^ab0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_8_layer_call_fn_11622891Qab0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_9_layer_call_and_return_conditional_losses_11622941]mn0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????	
? ~
*__inference_dense_9_layer_call_fn_11622950Pmn0?-
&?#
!?
inputs??????????
? "??????????	?
G__inference_flatten_1_layer_call_and_return_conditional_losses_11622630a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????}
? ?
,__inference_flatten_1_layer_call_fn_11622635T7?4
-?*
(?%
inputs????????? 
? "???????????}?
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_11621552?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_1_layer_call_fn_11621558?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_model_layer_call_and_return_conditional_losses_11622004{ !12=>IJUVabmn@?=
6?3
)?&
input_2?????????(2
p

 
? "%?"
?
0?????????	
? ?
C__inference_model_layer_call_and_return_conditional_losses_11622055{ !12=>IJUVabmn@?=
6?3
)?&
input_2?????????(2
p 

 
? "%?"
?
0?????????	
? ?
C__inference_model_layer_call_and_return_conditional_losses_11622442z !12=>IJUVabmn??<
5?2
(?%
inputs?????????(2
p

 
? "%?"
?
0?????????	
? ?
C__inference_model_layer_call_and_return_conditional_losses_11622510z !12=>IJUVabmn??<
5?2
(?%
inputs?????????(2
p 

 
? "%?"
?
0?????????	
? ?
(__inference_model_layer_call_fn_11622144n !12=>IJUVabmn@?=
6?3
)?&
input_2?????????(2
p

 
? "??????????	?
(__inference_model_layer_call_fn_11622232n !12=>IJUVabmn@?=
6?3
)?&
input_2?????????(2
p 

 
? "??????????	?
(__inference_model_layer_call_fn_11622547m !12=>IJUVabmn??<
5?2
(?%
inputs?????????(2
p

 
? "??????????	?
(__inference_model_layer_call_fn_11622584m !12=>IJUVabmn??<
5?2
(?%
inputs?????????(2
p 

 
? "??????????	?
&__inference_signature_wrapper_11622279? !12=>IJUVabmnC?@
? 
9?6
4
input_2)?&
input_2?????????(2"1?.
,
dense_9!?
dense_9?????????	