??
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
 ?"serve*2.4.12unknown8??
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
: *
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
: *
dtype0
?
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
: *
dtype0
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?d?* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
?d?*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:?*
dtype0
|
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_17/kernel
u
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel* 
_output_shapes
:
??*
dtype0
s
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_17/bias
l
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes	
:?*
dtype0
|
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_18/kernel
u
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel* 
_output_shapes
:
??*
dtype0
s
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes	
:?*
dtype0
{
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_19/kernel
t
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes
:	?*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
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
y
true_positives_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nametrue_positives_3
r
$true_positives_3/Read/ReadVariableOpReadVariableOptrue_positives_3*
_output_shapes	
:?*
dtype0
y
true_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nametrue_negatives_1
r
$true_negatives_1/Read/ReadVariableOpReadVariableOptrue_negatives_1*
_output_shapes	
:?*
dtype0
{
false_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_positives_2
t
%false_positives_2/Read/ReadVariableOpReadVariableOpfalse_positives_2*
_output_shapes	
:?*
dtype0
{
false_negatives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_negatives_2
t
%false_negatives_2/Read/ReadVariableOpReadVariableOpfalse_negatives_2*
_output_shapes	
:?*
dtype0
?
Nadam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameNadam/conv2d_8/kernel/m
?
+Nadam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_8/kernel/m*&
_output_shapes
: *
dtype0
?
Nadam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_8/bias/m
{
)Nadam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_8/bias/m*
_output_shapes
: *
dtype0
?
Nadam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameNadam/conv2d_9/kernel/m
?
+Nadam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_9/kernel/m*&
_output_shapes
:  *
dtype0
?
Nadam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_9/bias/m
{
)Nadam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_9/bias/m*
_output_shapes
: *
dtype0
?
Nadam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?d?*(
shared_nameNadam/dense_16/kernel/m
?
+Nadam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_16/kernel/m* 
_output_shapes
:
?d?*
dtype0
?
Nadam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_16/bias/m
|
)Nadam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_16/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_17/kernel/m
?
+Nadam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_17/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_17/bias/m
|
)Nadam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_17/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_18/kernel/m
?
+Nadam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_18/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_18/bias/m
|
)Nadam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_18/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameNadam/dense_19/kernel/m
?
+Nadam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_19/kernel/m*
_output_shapes
:	?*
dtype0
?
Nadam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_19/bias/m
{
)Nadam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_19/bias/m*
_output_shapes
:*
dtype0
?
Nadam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameNadam/conv2d_8/kernel/v
?
+Nadam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_8/kernel/v*&
_output_shapes
: *
dtype0
?
Nadam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_8/bias/v
{
)Nadam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_8/bias/v*
_output_shapes
: *
dtype0
?
Nadam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameNadam/conv2d_9/kernel/v
?
+Nadam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_9/kernel/v*&
_output_shapes
:  *
dtype0
?
Nadam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_9/bias/v
{
)Nadam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_9/bias/v*
_output_shapes
: *
dtype0
?
Nadam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?d?*(
shared_nameNadam/dense_16/kernel/v
?
+Nadam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_16/kernel/v* 
_output_shapes
:
?d?*
dtype0
?
Nadam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_16/bias/v
|
)Nadam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_16/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_17/kernel/v
?
+Nadam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_17/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_17/bias/v
|
)Nadam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_17/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_18/kernel/v
?
+Nadam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_18/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_18/bias/v
|
)Nadam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_18/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameNadam/dense_19/kernel/v
?
+Nadam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_19/kernel/v*
_output_shapes
:	?*
dtype0
?
Nadam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_19/bias/v
{
)Nadam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_19/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?[
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?Z
value?ZB?Z B?Z
?
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
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
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
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
R
7trainable_variables
8	variables
9regularization_losses
:	keras_api
h

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
R
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
h

Ekernel
Fbias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
?
Kiter

Lbeta_1

Mbeta_2
	Ndecay
Olearning_rate
Pmomentum_cachem?m?m?m?'m?(m?1m?2m?;m?<m?Em?Fm?v?v?v?v?'v?(v?1v?2v?;v?<v?Ev?Fv?
V
0
1
2
3
'4
(5
16
27
;8
<9
E10
F11
V
0
1
2
3
'4
(5
16
27
;8
<9
E10
F11
 
?
trainable_variables

Qlayers
	variables
Rnon_trainable_variables
regularization_losses
Slayer_regularization_losses
Tmetrics
Ulayer_metrics
 
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables

Vlayers
	variables
Wnon_trainable_variables
regularization_losses
Xlayer_regularization_losses
Ymetrics
Zlayer_metrics
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables

[layers
	variables
\non_trainable_variables
regularization_losses
]layer_regularization_losses
^metrics
_layer_metrics
 
 
 
?
trainable_variables

`layers
 	variables
anon_trainable_variables
!regularization_losses
blayer_regularization_losses
cmetrics
dlayer_metrics
 
 
 
?
#trainable_variables

elayers
$	variables
fnon_trainable_variables
%regularization_losses
glayer_regularization_losses
hmetrics
ilayer_metrics
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
)trainable_variables

jlayers
*	variables
knon_trainable_variables
+regularization_losses
llayer_regularization_losses
mmetrics
nlayer_metrics
 
 
 
?
-trainable_variables

olayers
.	variables
pnon_trainable_variables
/regularization_losses
qlayer_regularization_losses
rmetrics
slayer_metrics
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
3trainable_variables

tlayers
4	variables
unon_trainable_variables
5regularization_losses
vlayer_regularization_losses
wmetrics
xlayer_metrics
 
 
 
?
7trainable_variables

ylayers
8	variables
znon_trainable_variables
9regularization_losses
{layer_regularization_losses
|metrics
}layer_metrics
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
?
=trainable_variables

~layers
>	variables
non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
 
 
 
?
Atrainable_variables
?layers
B	variables
?non_trainable_variables
Cregularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

E0
F1
 
?
Gtrainable_variables
?layers
H	variables
?non_trainable_variables
Iregularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
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
V
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
 
 
0
?0
?1
?2
?3
?4
?5
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
?
?true_positives
?true_negatives
?false_positives
?false_negatives
?
thresholds
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
ca
VARIABLE_VALUEtrue_positives_3=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtrue_negatives_1=keras_api/metrics/5/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_2>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_2>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
 
?0
?1
?2
?3

?	variables
}
VARIABLE_VALUENadam/conv2d_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv2d_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv2d_9/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv2d_9/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_16/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_16/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_17/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_17/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_18/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_18/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_19/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_19/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv2d_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv2d_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv2d_9/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv2d_9/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_16/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_16/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_17/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_17/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_18/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_18/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_19/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_19/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_5Placeholder*/
_output_shapes
:?????????((*
dtype0*$
shape:?????????((
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5conv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *-
f(R&
$__inference_signature_wrapper_963964
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp$true_positives_3/Read/ReadVariableOp$true_negatives_1/Read/ReadVariableOp%false_positives_2/Read/ReadVariableOp%false_negatives_2/Read/ReadVariableOp+Nadam/conv2d_8/kernel/m/Read/ReadVariableOp)Nadam/conv2d_8/bias/m/Read/ReadVariableOp+Nadam/conv2d_9/kernel/m/Read/ReadVariableOp)Nadam/conv2d_9/bias/m/Read/ReadVariableOp+Nadam/dense_16/kernel/m/Read/ReadVariableOp)Nadam/dense_16/bias/m/Read/ReadVariableOp+Nadam/dense_17/kernel/m/Read/ReadVariableOp)Nadam/dense_17/bias/m/Read/ReadVariableOp+Nadam/dense_18/kernel/m/Read/ReadVariableOp)Nadam/dense_18/bias/m/Read/ReadVariableOp+Nadam/dense_19/kernel/m/Read/ReadVariableOp)Nadam/dense_19/bias/m/Read/ReadVariableOp+Nadam/conv2d_8/kernel/v/Read/ReadVariableOp)Nadam/conv2d_8/bias/v/Read/ReadVariableOp+Nadam/conv2d_9/kernel/v/Read/ReadVariableOp)Nadam/conv2d_9/bias/v/Read/ReadVariableOp+Nadam/dense_16/kernel/v/Read/ReadVariableOp)Nadam/dense_16/bias/v/Read/ReadVariableOp+Nadam/dense_17/kernel/v/Read/ReadVariableOp)Nadam/dense_17/bias/v/Read/ReadVariableOp+Nadam/dense_18/kernel/v/Read/ReadVariableOp)Nadam/dense_18/bias/v/Read/ReadVariableOp+Nadam/dense_19/kernel/v/Read/ReadVariableOp)Nadam/dense_19/bias/v/Read/ReadVariableOpConst*G
Tin@
>2<	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *(
f#R!
__inference__traced_save_964628
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcounttotal_1count_1true_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1false_positives_1true_positives_2false_negatives_1true_positives_3true_negatives_1false_positives_2false_negatives_2Nadam/conv2d_8/kernel/mNadam/conv2d_8/bias/mNadam/conv2d_9/kernel/mNadam/conv2d_9/bias/mNadam/dense_16/kernel/mNadam/dense_16/bias/mNadam/dense_17/kernel/mNadam/dense_17/bias/mNadam/dense_18/kernel/mNadam/dense_18/bias/mNadam/dense_19/kernel/mNadam/dense_19/bias/mNadam/conv2d_8/kernel/vNadam/conv2d_8/bias/vNadam/conv2d_9/kernel/vNadam/conv2d_9/bias/vNadam/dense_16/kernel/vNadam/dense_16/bias/vNadam/dense_17/kernel/vNadam/dense_17/bias/vNadam/dense_18/kernel/vNadam/dense_18/bias/vNadam/dense_19/kernel/vNadam/dense_19/bias/v*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *+
f&R$
"__inference__traced_restore_964812??	
?A
?
C__inference_model_4_layer_call_and_return_conditional_losses_964125

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource
identity??conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_8/Relu?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingSAME*
strides
2
conv2d_9/Conv2D?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_9/Relu?
max_pooling2d_4/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPools
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 2  2
flatten_4/Const?
flatten_4/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????d2
flatten_4/Reshape?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
?d?*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMulflatten_4/Reshape:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/BiasAddt
dense_16/SeluSeludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_16/Selu{
alpha_dropout_12/ShapeShapedense_16/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_12/Shape?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_16/Selu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/BiasAddt
dense_17/SeluSeludense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_17/Selu{
alpha_dropout_13/ShapeShapedense_17/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_13/Shape?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_18/MatMul/ReadVariableOp?
dense_18/MatMulMatMuldense_17/Selu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/MatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/BiasAddt
dense_18/SeluSeludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_18/Selu{
alpha_dropout_14/ShapeShapedense_18/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_14/Shape?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_19/MatMul/ReadVariableOp?
dense_19/MatMulMatMuldense_18/Selu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/MatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/BiasAdd|
dense_19/SigmoidSigmoiddense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_19/Sigmoid?
IdentityIdentitydense_19/Sigmoid:y:0 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
?	
?
(__inference_model_4_layer_call_fn_964183

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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_9638982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
?6
?
C__inference_model_4_layer_call_and_return_conditional_losses_963830

inputs
conv2d_8_963794
conv2d_8_963796
conv2d_9_963799
conv2d_9_963801
dense_16_963806
dense_16_963808
dense_17_963812
dense_17_963814
dense_18_963818
dense_18_963820
dense_19_963824
dense_19_963826
identity??(alpha_dropout_12/StatefulPartitionedCall?(alpha_dropout_13/StatefulPartitionedCall?(alpha_dropout_14/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_963794conv2d_8_963796*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_9634562"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_963799conv2d_9_963801*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_9634832"
 conv2d_9/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_9634352!
max_pooling2d_4/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????d* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_9635062
flatten_4/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_16_963806dense_16_963808*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_9635252"
 dense_16/StatefulPartitionedCall?
(alpha_dropout_12/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_9635652*
(alpha_dropout_12/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall1alpha_dropout_12/StatefulPartitionedCall:output:0dense_17_963812dense_17_963814*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_9635942"
 dense_17/StatefulPartitionedCall?
(alpha_dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0)^alpha_dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_9636342*
(alpha_dropout_13/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall1alpha_dropout_13/StatefulPartitionedCall:output:0dense_18_963818dense_18_963820*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_9636632"
 dense_18/StatefulPartitionedCall?
(alpha_dropout_14/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0)^alpha_dropout_13/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_9637032*
(alpha_dropout_14/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall1alpha_dropout_14/StatefulPartitionedCall:output:0dense_19_963824dense_19_963826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_9637322"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0)^alpha_dropout_12/StatefulPartitionedCall)^alpha_dropout_13/StatefulPartitionedCall)^alpha_dropout_14/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::2T
(alpha_dropout_12/StatefulPartitionedCall(alpha_dropout_12/StatefulPartitionedCall2T
(alpha_dropout_13/StatefulPartitionedCall(alpha_dropout_13/StatefulPartitionedCall2T
(alpha_dropout_14/StatefulPartitionedCall(alpha_dropout_14/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
ދ
?
C__inference_model_4_layer_call_and_return_conditional_losses_964073

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource
identity??conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_8/Relu?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingSAME*
strides
2
conv2d_9/Conv2D?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_9/Relu?
max_pooling2d_4/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPools
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 2  2
flatten_4/Const?
flatten_4/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????d2
flatten_4/Reshape?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
?d?*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMulflatten_4/Reshape:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/BiasAddt
dense_16/SeluSeludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_16/Selu{
alpha_dropout_12/ShapeShapedense_16/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_12/Shape?
#alpha_dropout_12/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#alpha_dropout_12/random_uniform/min?
#alpha_dropout_12/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#alpha_dropout_12/random_uniform/max?
-alpha_dropout_12/random_uniform/RandomUniformRandomUniformalpha_dropout_12/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2/
-alpha_dropout_12/random_uniform/RandomUniform?
#alpha_dropout_12/random_uniform/subSub,alpha_dropout_12/random_uniform/max:output:0,alpha_dropout_12/random_uniform/min:output:0*
T0*
_output_shapes
: 2%
#alpha_dropout_12/random_uniform/sub?
#alpha_dropout_12/random_uniform/mulMul6alpha_dropout_12/random_uniform/RandomUniform:output:0'alpha_dropout_12/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2%
#alpha_dropout_12/random_uniform/mul?
alpha_dropout_12/random_uniformAdd'alpha_dropout_12/random_uniform/mul:z:0,alpha_dropout_12/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2!
alpha_dropout_12/random_uniform?
alpha_dropout_12/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
alpha_dropout_12/GreaterEqual/y?
alpha_dropout_12/GreaterEqualGreaterEqual#alpha_dropout_12/random_uniform:z:0(alpha_dropout_12/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_12/GreaterEqual?
alpha_dropout_12/CastCast!alpha_dropout_12/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
alpha_dropout_12/Cast?
alpha_dropout_12/mulMuldense_16/Selu:activations:0alpha_dropout_12/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_12/mulu
alpha_dropout_12/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
alpha_dropout_12/sub/x?
alpha_dropout_12/subSubalpha_dropout_12/sub/x:output:0alpha_dropout_12/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_12/suby
alpha_dropout_12/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2
alpha_dropout_12/mul_1/x?
alpha_dropout_12/mul_1Mul!alpha_dropout_12/mul_1/x:output:0alpha_dropout_12/sub:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_12/mul_1?
alpha_dropout_12/addAddV2alpha_dropout_12/mul:z:0alpha_dropout_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_12/addy
alpha_dropout_12/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2
alpha_dropout_12/mul_2/x?
alpha_dropout_12/mul_2Mul!alpha_dropout_12/mul_2/x:output:0alpha_dropout_12/add:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_12/mul_2y
alpha_dropout_12/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2
alpha_dropout_12/add_1/y?
alpha_dropout_12/add_1AddV2alpha_dropout_12/mul_2:z:0!alpha_dropout_12/add_1/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_12/add_1?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMulalpha_dropout_12/add_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/BiasAddt
dense_17/SeluSeludense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_17/Selu{
alpha_dropout_13/ShapeShapedense_17/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_13/Shape?
#alpha_dropout_13/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#alpha_dropout_13/random_uniform/min?
#alpha_dropout_13/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#alpha_dropout_13/random_uniform/max?
-alpha_dropout_13/random_uniform/RandomUniformRandomUniformalpha_dropout_13/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??y2/
-alpha_dropout_13/random_uniform/RandomUniform?
#alpha_dropout_13/random_uniform/subSub,alpha_dropout_13/random_uniform/max:output:0,alpha_dropout_13/random_uniform/min:output:0*
T0*
_output_shapes
: 2%
#alpha_dropout_13/random_uniform/sub?
#alpha_dropout_13/random_uniform/mulMul6alpha_dropout_13/random_uniform/RandomUniform:output:0'alpha_dropout_13/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2%
#alpha_dropout_13/random_uniform/mul?
alpha_dropout_13/random_uniformAdd'alpha_dropout_13/random_uniform/mul:z:0,alpha_dropout_13/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2!
alpha_dropout_13/random_uniform?
alpha_dropout_13/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
alpha_dropout_13/GreaterEqual/y?
alpha_dropout_13/GreaterEqualGreaterEqual#alpha_dropout_13/random_uniform:z:0(alpha_dropout_13/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_13/GreaterEqual?
alpha_dropout_13/CastCast!alpha_dropout_13/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
alpha_dropout_13/Cast?
alpha_dropout_13/mulMuldense_17/Selu:activations:0alpha_dropout_13/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_13/mulu
alpha_dropout_13/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
alpha_dropout_13/sub/x?
alpha_dropout_13/subSubalpha_dropout_13/sub/x:output:0alpha_dropout_13/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_13/suby
alpha_dropout_13/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2
alpha_dropout_13/mul_1/x?
alpha_dropout_13/mul_1Mul!alpha_dropout_13/mul_1/x:output:0alpha_dropout_13/sub:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_13/mul_1?
alpha_dropout_13/addAddV2alpha_dropout_13/mul:z:0alpha_dropout_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_13/addy
alpha_dropout_13/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2
alpha_dropout_13/mul_2/x?
alpha_dropout_13/mul_2Mul!alpha_dropout_13/mul_2/x:output:0alpha_dropout_13/add:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_13/mul_2y
alpha_dropout_13/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2
alpha_dropout_13/add_1/y?
alpha_dropout_13/add_1AddV2alpha_dropout_13/mul_2:z:0!alpha_dropout_13/add_1/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_13/add_1?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_18/MatMul/ReadVariableOp?
dense_18/MatMulMatMulalpha_dropout_13/add_1:z:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/MatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/BiasAddt
dense_18/SeluSeludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_18/Selu{
alpha_dropout_14/ShapeShapedense_18/Selu:activations:0*
T0*
_output_shapes
:2
alpha_dropout_14/Shape?
#alpha_dropout_14/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#alpha_dropout_14/random_uniform/min?
#alpha_dropout_14/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#alpha_dropout_14/random_uniform/max?
-alpha_dropout_14/random_uniform/RandomUniformRandomUniformalpha_dropout_14/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ݝ?2/
-alpha_dropout_14/random_uniform/RandomUniform?
#alpha_dropout_14/random_uniform/subSub,alpha_dropout_14/random_uniform/max:output:0,alpha_dropout_14/random_uniform/min:output:0*
T0*
_output_shapes
: 2%
#alpha_dropout_14/random_uniform/sub?
#alpha_dropout_14/random_uniform/mulMul6alpha_dropout_14/random_uniform/RandomUniform:output:0'alpha_dropout_14/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2%
#alpha_dropout_14/random_uniform/mul?
alpha_dropout_14/random_uniformAdd'alpha_dropout_14/random_uniform/mul:z:0,alpha_dropout_14/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2!
alpha_dropout_14/random_uniform?
alpha_dropout_14/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
alpha_dropout_14/GreaterEqual/y?
alpha_dropout_14/GreaterEqualGreaterEqual#alpha_dropout_14/random_uniform:z:0(alpha_dropout_14/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_14/GreaterEqual?
alpha_dropout_14/CastCast!alpha_dropout_14/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
alpha_dropout_14/Cast?
alpha_dropout_14/mulMuldense_18/Selu:activations:0alpha_dropout_14/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_14/mulu
alpha_dropout_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
alpha_dropout_14/sub/x?
alpha_dropout_14/subSubalpha_dropout_14/sub/x:output:0alpha_dropout_14/Cast:y:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_14/suby
alpha_dropout_14/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *f	??2
alpha_dropout_14/mul_1/x?
alpha_dropout_14/mul_1Mul!alpha_dropout_14/mul_1/x:output:0alpha_dropout_14/sub:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_14/mul_1?
alpha_dropout_14/addAddV2alpha_dropout_14/mul:z:0alpha_dropout_14/mul_1:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_14/addy
alpha_dropout_14/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *??`?2
alpha_dropout_14/mul_2/x?
alpha_dropout_14/mul_2Mul!alpha_dropout_14/mul_2/x:output:0alpha_dropout_14/add:z:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_14/mul_2y
alpha_dropout_14/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *|:?>2
alpha_dropout_14/add_1/y?
alpha_dropout_14/add_1AddV2alpha_dropout_14/mul_2:z:0!alpha_dropout_14/add_1/y:output:0*
T0*(
_output_shapes
:??????????2
alpha_dropout_14/add_1?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_19/MatMul/ReadVariableOp?
dense_19/MatMulMatMulalpha_dropout_14/add_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/MatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/BiasAdd|
dense_19/SigmoidSigmoiddense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_19/Sigmoid?
IdentityIdentitydense_19/Sigmoid:y:0 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
?
k
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_963565

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
?
j
1__inference_alpha_dropout_13_layer_call_fn_964347

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
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_9636342
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
h
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_964342

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
?
M
1__inference_alpha_dropout_14_layer_call_fn_964411

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
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_9637082
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
j
1__inference_alpha_dropout_14_layer_call_fn_964406

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
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_9637032
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
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_964229

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 2  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????d2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?L
?	
!__inference__wrapped_model_963429
input_53
/model_4_conv2d_8_conv2d_readvariableop_resource4
0model_4_conv2d_8_biasadd_readvariableop_resource3
/model_4_conv2d_9_conv2d_readvariableop_resource4
0model_4_conv2d_9_biasadd_readvariableop_resource3
/model_4_dense_16_matmul_readvariableop_resource4
0model_4_dense_16_biasadd_readvariableop_resource3
/model_4_dense_17_matmul_readvariableop_resource4
0model_4_dense_17_biasadd_readvariableop_resource3
/model_4_dense_18_matmul_readvariableop_resource4
0model_4_dense_18_biasadd_readvariableop_resource3
/model_4_dense_19_matmul_readvariableop_resource4
0model_4_dense_19_biasadd_readvariableop_resource
identity??'model_4/conv2d_8/BiasAdd/ReadVariableOp?&model_4/conv2d_8/Conv2D/ReadVariableOp?'model_4/conv2d_9/BiasAdd/ReadVariableOp?&model_4/conv2d_9/Conv2D/ReadVariableOp?'model_4/dense_16/BiasAdd/ReadVariableOp?&model_4/dense_16/MatMul/ReadVariableOp?'model_4/dense_17/BiasAdd/ReadVariableOp?&model_4/dense_17/MatMul/ReadVariableOp?'model_4/dense_18/BiasAdd/ReadVariableOp?&model_4/dense_18/MatMul/ReadVariableOp?'model_4/dense_19/BiasAdd/ReadVariableOp?&model_4/dense_19/MatMul/ReadVariableOp?
&model_4/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/model_4_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&model_4/conv2d_8/Conv2D/ReadVariableOp?
model_4/conv2d_8/Conv2DConv2Dinput_5.model_4/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingSAME*
strides
2
model_4/conv2d_8/Conv2D?
'model_4/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0model_4_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_4/conv2d_8/BiasAdd/ReadVariableOp?
model_4/conv2d_8/BiasAddBiasAdd model_4/conv2d_8/Conv2D:output:0/model_4/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2
model_4/conv2d_8/BiasAdd?
model_4/conv2d_8/ReluRelu!model_4/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
model_4/conv2d_8/Relu?
&model_4/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/model_4_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02(
&model_4/conv2d_9/Conv2D/ReadVariableOp?
model_4/conv2d_9/Conv2DConv2D#model_4/conv2d_8/Relu:activations:0.model_4/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingSAME*
strides
2
model_4/conv2d_9/Conv2D?
'model_4/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0model_4_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_4/conv2d_9/BiasAdd/ReadVariableOp?
model_4/conv2d_9/BiasAddBiasAdd model_4/conv2d_9/Conv2D:output:0/model_4/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2
model_4/conv2d_9/BiasAdd?
model_4/conv2d_9/ReluRelu!model_4/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
model_4/conv2d_9/Relu?
model_4/max_pooling2d_4/MaxPoolMaxPool#model_4/conv2d_9/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2!
model_4/max_pooling2d_4/MaxPool?
model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 2  2
model_4/flatten_4/Const?
model_4/flatten_4/ReshapeReshape(model_4/max_pooling2d_4/MaxPool:output:0 model_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????d2
model_4/flatten_4/Reshape?
&model_4/dense_16/MatMul/ReadVariableOpReadVariableOp/model_4_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
?d?*
dtype02(
&model_4/dense_16/MatMul/ReadVariableOp?
model_4/dense_16/MatMulMatMul"model_4/flatten_4/Reshape:output:0.model_4/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_16/MatMul?
'model_4/dense_16/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_4/dense_16/BiasAdd/ReadVariableOp?
model_4/dense_16/BiasAddBiasAdd!model_4/dense_16/MatMul:product:0/model_4/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_16/BiasAdd?
model_4/dense_16/SeluSelu!model_4/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_4/dense_16/Selu?
model_4/alpha_dropout_12/ShapeShape#model_4/dense_16/Selu:activations:0*
T0*
_output_shapes
:2 
model_4/alpha_dropout_12/Shape?
&model_4/dense_17/MatMul/ReadVariableOpReadVariableOp/model_4_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&model_4/dense_17/MatMul/ReadVariableOp?
model_4/dense_17/MatMulMatMul#model_4/dense_16/Selu:activations:0.model_4/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_17/MatMul?
'model_4/dense_17/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_4/dense_17/BiasAdd/ReadVariableOp?
model_4/dense_17/BiasAddBiasAdd!model_4/dense_17/MatMul:product:0/model_4/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_17/BiasAdd?
model_4/dense_17/SeluSelu!model_4/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_4/dense_17/Selu?
model_4/alpha_dropout_13/ShapeShape#model_4/dense_17/Selu:activations:0*
T0*
_output_shapes
:2 
model_4/alpha_dropout_13/Shape?
&model_4/dense_18/MatMul/ReadVariableOpReadVariableOp/model_4_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&model_4/dense_18/MatMul/ReadVariableOp?
model_4/dense_18/MatMulMatMul#model_4/dense_17/Selu:activations:0.model_4/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_18/MatMul?
'model_4/dense_18/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_4/dense_18/BiasAdd/ReadVariableOp?
model_4/dense_18/BiasAddBiasAdd!model_4/dense_18/MatMul:product:0/model_4/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense_18/BiasAdd?
model_4/dense_18/SeluSelu!model_4/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_4/dense_18/Selu?
model_4/alpha_dropout_14/ShapeShape#model_4/dense_18/Selu:activations:0*
T0*
_output_shapes
:2 
model_4/alpha_dropout_14/Shape?
&model_4/dense_19/MatMul/ReadVariableOpReadVariableOp/model_4_dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_4/dense_19/MatMul/ReadVariableOp?
model_4/dense_19/MatMulMatMul#model_4/dense_18/Selu:activations:0.model_4/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/dense_19/MatMul?
'model_4/dense_19/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_4/dense_19/BiasAdd/ReadVariableOp?
model_4/dense_19/BiasAddBiasAdd!model_4/dense_19/MatMul:product:0/model_4/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/dense_19/BiasAdd?
model_4/dense_19/SigmoidSigmoid!model_4/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/dense_19/Sigmoid?
IdentityIdentitymodel_4/dense_19/Sigmoid:y:0(^model_4/conv2d_8/BiasAdd/ReadVariableOp'^model_4/conv2d_8/Conv2D/ReadVariableOp(^model_4/conv2d_9/BiasAdd/ReadVariableOp'^model_4/conv2d_9/Conv2D/ReadVariableOp(^model_4/dense_16/BiasAdd/ReadVariableOp'^model_4/dense_16/MatMul/ReadVariableOp(^model_4/dense_17/BiasAdd/ReadVariableOp'^model_4/dense_17/MatMul/ReadVariableOp(^model_4/dense_18/BiasAdd/ReadVariableOp'^model_4/dense_18/MatMul/ReadVariableOp(^model_4/dense_19/BiasAdd/ReadVariableOp'^model_4/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::2R
'model_4/conv2d_8/BiasAdd/ReadVariableOp'model_4/conv2d_8/BiasAdd/ReadVariableOp2P
&model_4/conv2d_8/Conv2D/ReadVariableOp&model_4/conv2d_8/Conv2D/ReadVariableOp2R
'model_4/conv2d_9/BiasAdd/ReadVariableOp'model_4/conv2d_9/BiasAdd/ReadVariableOp2P
&model_4/conv2d_9/Conv2D/ReadVariableOp&model_4/conv2d_9/Conv2D/ReadVariableOp2R
'model_4/dense_16/BiasAdd/ReadVariableOp'model_4/dense_16/BiasAdd/ReadVariableOp2P
&model_4/dense_16/MatMul/ReadVariableOp&model_4/dense_16/MatMul/ReadVariableOp2R
'model_4/dense_17/BiasAdd/ReadVariableOp'model_4/dense_17/BiasAdd/ReadVariableOp2P
&model_4/dense_17/MatMul/ReadVariableOp&model_4/dense_17/MatMul/ReadVariableOp2R
'model_4/dense_18/BiasAdd/ReadVariableOp'model_4/dense_18/BiasAdd/ReadVariableOp2P
&model_4/dense_18/MatMul/ReadVariableOp&model_4/dense_18/MatMul/ReadVariableOp2R
'model_4/dense_19/BiasAdd/ReadVariableOp'model_4/dense_19/BiasAdd/ReadVariableOp2P
&model_4/dense_19/MatMul/ReadVariableOp&model_4/dense_19/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_5
?
~
)__inference_conv2d_9_layer_call_fn_964223

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
:?????????(( *$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_9634832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(( 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(( ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(( 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_964214

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
:?????????(( *
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
:?????????(( 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(( 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(( ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(( 
 
_user_specified_nameinputs
?
F
*__inference_flatten_4_layer_call_fn_964234

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
:??????????d* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_9635062
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_963506

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 2  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????d2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_18_layer_call_and_return_conditional_losses_963663

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
k
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_963634

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
seed2Đ?2
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
D__inference_dense_19_layer_call_and_return_conditional_losses_964422

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
?
"__inference__traced_restore_964812
file_prefix$
 assignvariableop_conv2d_8_kernel$
 assignvariableop_1_conv2d_8_bias&
"assignvariableop_2_conv2d_9_kernel$
 assignvariableop_3_conv2d_9_bias&
"assignvariableop_4_dense_16_kernel$
 assignvariableop_5_dense_16_bias&
"assignvariableop_6_dense_17_kernel$
 assignvariableop_7_dense_17_bias&
"assignvariableop_8_dense_18_kernel$
 assignvariableop_9_dense_18_bias'
#assignvariableop_10_dense_19_kernel%
!assignvariableop_11_dense_19_bias"
assignvariableop_12_nadam_iter$
 assignvariableop_13_nadam_beta_1$
 assignvariableop_14_nadam_beta_2#
assignvariableop_15_nadam_decay+
'assignvariableop_16_nadam_learning_rate,
(assignvariableop_17_nadam_momentum_cache
assignvariableop_18_total
assignvariableop_19_count
assignvariableop_20_total_1
assignvariableop_21_count_1&
"assignvariableop_22_true_positives&
"assignvariableop_23_true_negatives'
#assignvariableop_24_false_positives'
#assignvariableop_25_false_negatives(
$assignvariableop_26_true_positives_1)
%assignvariableop_27_false_positives_1(
$assignvariableop_28_true_positives_2)
%assignvariableop_29_false_negatives_1(
$assignvariableop_30_true_positives_3(
$assignvariableop_31_true_negatives_1)
%assignvariableop_32_false_positives_2)
%assignvariableop_33_false_negatives_2/
+assignvariableop_34_nadam_conv2d_8_kernel_m-
)assignvariableop_35_nadam_conv2d_8_bias_m/
+assignvariableop_36_nadam_conv2d_9_kernel_m-
)assignvariableop_37_nadam_conv2d_9_bias_m/
+assignvariableop_38_nadam_dense_16_kernel_m-
)assignvariableop_39_nadam_dense_16_bias_m/
+assignvariableop_40_nadam_dense_17_kernel_m-
)assignvariableop_41_nadam_dense_17_bias_m/
+assignvariableop_42_nadam_dense_18_kernel_m-
)assignvariableop_43_nadam_dense_18_bias_m/
+assignvariableop_44_nadam_dense_19_kernel_m-
)assignvariableop_45_nadam_dense_19_bias_m/
+assignvariableop_46_nadam_conv2d_8_kernel_v-
)assignvariableop_47_nadam_conv2d_8_bias_v/
+assignvariableop_48_nadam_conv2d_9_kernel_v-
)assignvariableop_49_nadam_conv2d_9_bias_v/
+assignvariableop_50_nadam_dense_16_kernel_v-
)assignvariableop_51_nadam_dense_16_bias_v/
+assignvariableop_52_nadam_dense_17_kernel_v-
)assignvariableop_53_nadam_dense_17_bias_v/
+assignvariableop_54_nadam_dense_18_kernel_v-
)assignvariableop_55_nadam_dense_18_bias_v/
+assignvariableop_56_nadam_dense_19_kernel_v-
)assignvariableop_57_nadam_dense_19_bias_v
identity_59??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B?;B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_16_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_16_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_17_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_17_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_18_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_18_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_19_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_19_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_nadam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_nadam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp assignvariableop_14_nadam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_nadam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_nadam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp(assignvariableop_17_nadam_momentum_cacheIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_true_positivesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp"assignvariableop_23_true_negativesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_positivesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp#assignvariableop_25_false_negativesIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_true_positives_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_false_positives_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp$assignvariableop_28_true_positives_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp%assignvariableop_29_false_negatives_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp$assignvariableop_30_true_positives_3Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp$assignvariableop_31_true_negatives_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_false_positives_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_false_negatives_2Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_nadam_conv2d_8_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_nadam_conv2d_8_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_nadam_conv2d_9_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_nadam_conv2d_9_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp+assignvariableop_38_nadam_dense_16_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_nadam_dense_16_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp+assignvariableop_40_nadam_dense_17_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_nadam_dense_17_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp+assignvariableop_42_nadam_dense_18_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_nadam_dense_18_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp+assignvariableop_44_nadam_dense_19_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_nadam_dense_19_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp+assignvariableop_46_nadam_conv2d_8_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp)assignvariableop_47_nadam_conv2d_8_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp+assignvariableop_48_nadam_conv2d_9_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp)assignvariableop_49_nadam_conv2d_9_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp+assignvariableop_50_nadam_dense_16_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp)assignvariableop_51_nadam_dense_16_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp+assignvariableop_52_nadam_dense_17_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp)assignvariableop_53_nadam_dense_17_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp+assignvariableop_54_nadam_dense_18_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp)assignvariableop_55_nadam_dense_18_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp+assignvariableop_56_nadam_dense_19_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp)assignvariableop_57_nadam_dense_19_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_579
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_58Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_58?

Identity_59IdentityIdentity_58:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_59"#
identity_59Identity_59:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_57AssignVariableOp_572(
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

?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_964194

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
:?????????(( *
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
:?????????(( 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(( 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????((::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_963964
input_5
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? **
f%R#
!__inference__wrapped_model_9634292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_5
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_964304

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
D__inference_dense_17_layer_call_and_return_conditional_losses_963594

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
~
)__inference_dense_18_layer_call_fn_964372

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
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_9636632
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
?
M
1__inference_alpha_dropout_12_layer_call_fn_964293

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
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_9635702
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

?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_963456

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
:?????????(( *
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
:?????????(( 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(( 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????((::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
?	
?
(__inference_model_4_layer_call_fn_963925
input_5
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_9638982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_5
?
~
)__inference_dense_16_layer_call_fn_964254

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
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_9635252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????d::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????d
 
_user_specified_nameinputs
?	
?
D__inference_dense_16_layer_call_and_return_conditional_losses_963525

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?d?*
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
:??????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????d
 
_user_specified_nameinputs
?	
?
D__inference_dense_19_layer_call_and_return_conditional_losses_963732

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
?6
?
C__inference_model_4_layer_call_and_return_conditional_losses_963749
input_5
conv2d_8_963467
conv2d_8_963469
conv2d_9_963494
conv2d_9_963496
dense_16_963536
dense_16_963538
dense_17_963605
dense_17_963607
dense_18_963674
dense_18_963676
dense_19_963743
dense_19_963745
identity??(alpha_dropout_12/StatefulPartitionedCall?(alpha_dropout_13/StatefulPartitionedCall?(alpha_dropout_14/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_8_963467conv2d_8_963469*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_9634562"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_963494conv2d_9_963496*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_9634832"
 conv2d_9/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_9634352!
max_pooling2d_4/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????d* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_9635062
flatten_4/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_16_963536dense_16_963538*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_9635252"
 dense_16/StatefulPartitionedCall?
(alpha_dropout_12/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_9635652*
(alpha_dropout_12/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall1alpha_dropout_12/StatefulPartitionedCall:output:0dense_17_963605dense_17_963607*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_9635942"
 dense_17/StatefulPartitionedCall?
(alpha_dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0)^alpha_dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_9636342*
(alpha_dropout_13/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall1alpha_dropout_13/StatefulPartitionedCall:output:0dense_18_963674dense_18_963676*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_9636632"
 dense_18/StatefulPartitionedCall?
(alpha_dropout_14/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0)^alpha_dropout_13/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_9637032*
(alpha_dropout_14/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall1alpha_dropout_14/StatefulPartitionedCall:output:0dense_19_963743dense_19_963745*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_9637322"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0)^alpha_dropout_12/StatefulPartitionedCall)^alpha_dropout_13/StatefulPartitionedCall)^alpha_dropout_14/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::2T
(alpha_dropout_12/StatefulPartitionedCall(alpha_dropout_12/StatefulPartitionedCall2T
(alpha_dropout_13/StatefulPartitionedCall(alpha_dropout_13/StatefulPartitionedCall2T
(alpha_dropout_14/StatefulPartitionedCall(alpha_dropout_14/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_5
?
L
0__inference_max_pooling2d_4_layer_call_fn_963441

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
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_9634352
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
D__inference_conv2d_9_layer_call_and_return_conditional_losses_963483

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
:?????????(( *
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
:?????????(( 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(( 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(( ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(( 
 
_user_specified_nameinputs
?
M
1__inference_alpha_dropout_13_layer_call_fn_964352

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
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_9636392
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
g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_963435

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
?
~
)__inference_conv2d_8_layer_call_fn_964203

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
:?????????(( *$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_9634562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(( 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????((::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
?
k
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_964337

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
?
k
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_964396

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
seed2使2
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
~
)__inference_dense_17_layer_call_fn_964313

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
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_9635942
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
?1
?
C__inference_model_4_layer_call_and_return_conditional_losses_963898

inputs
conv2d_8_963862
conv2d_8_963864
conv2d_9_963867
conv2d_9_963869
dense_16_963874
dense_16_963876
dense_17_963880
dense_17_963882
dense_18_963886
dense_18_963888
dense_19_963892
dense_19_963894
identity?? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_963862conv2d_8_963864*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_9634562"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_963867conv2d_9_963869*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_9634832"
 conv2d_9/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_9634352!
max_pooling2d_4/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????d* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_9635062
flatten_4/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_16_963874dense_16_963876*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_9635252"
 dense_16/StatefulPartitionedCall?
 alpha_dropout_12/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_9635702"
 alpha_dropout_12/PartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)alpha_dropout_12/PartitionedCall:output:0dense_17_963880dense_17_963882*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_9635942"
 dense_17/StatefulPartitionedCall?
 alpha_dropout_13/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_9636392"
 alpha_dropout_13/PartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)alpha_dropout_13/PartitionedCall:output:0dense_18_963886dense_18_963888*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_9636632"
 dense_18/StatefulPartitionedCall?
 alpha_dropout_14/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_9637082"
 alpha_dropout_14/PartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)alpha_dropout_14/PartitionedCall:output:0dense_19_963892dense_19_963894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_9637322"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
?	
?
D__inference_dense_16_layer_call_and_return_conditional_losses_964245

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?d?*
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
:??????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????d
 
_user_specified_nameinputs
?
k
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_964278

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
seed2??C2
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
?r
?
__inference__traced_save_964628
file_prefix.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop)
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
,savev2_false_negatives_1_read_readvariableop/
+savev2_true_positives_3_read_readvariableop/
+savev2_true_negatives_1_read_readvariableop0
,savev2_false_positives_2_read_readvariableop0
,savev2_false_negatives_2_read_readvariableop6
2savev2_nadam_conv2d_8_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_8_bias_m_read_readvariableop6
2savev2_nadam_conv2d_9_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_9_bias_m_read_readvariableop6
2savev2_nadam_dense_16_kernel_m_read_readvariableop4
0savev2_nadam_dense_16_bias_m_read_readvariableop6
2savev2_nadam_dense_17_kernel_m_read_readvariableop4
0savev2_nadam_dense_17_bias_m_read_readvariableop6
2savev2_nadam_dense_18_kernel_m_read_readvariableop4
0savev2_nadam_dense_18_bias_m_read_readvariableop6
2savev2_nadam_dense_19_kernel_m_read_readvariableop4
0savev2_nadam_dense_19_bias_m_read_readvariableop6
2savev2_nadam_conv2d_8_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_8_bias_v_read_readvariableop6
2savev2_nadam_conv2d_9_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_9_bias_v_read_readvariableop6
2savev2_nadam_dense_16_kernel_v_read_readvariableop4
0savev2_nadam_dense_16_bias_v_read_readvariableop6
2savev2_nadam_dense_17_kernel_v_read_readvariableop4
0savev2_nadam_dense_17_bias_v_read_readvariableop6
2savev2_nadam_dense_18_kernel_v_read_readvariableop4
0savev2_nadam_dense_18_bias_v_read_readvariableop6
2savev2_nadam_dense_19_kernel_v_read_readvariableop4
0savev2_nadam_dense_19_bias_v_read_readvariableop
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
:;*
dtype0*?
value?B?;B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop+savev2_true_positives_2_read_readvariableop,savev2_false_negatives_1_read_readvariableop+savev2_true_positives_3_read_readvariableop+savev2_true_negatives_1_read_readvariableop,savev2_false_positives_2_read_readvariableop,savev2_false_negatives_2_read_readvariableop2savev2_nadam_conv2d_8_kernel_m_read_readvariableop0savev2_nadam_conv2d_8_bias_m_read_readvariableop2savev2_nadam_conv2d_9_kernel_m_read_readvariableop0savev2_nadam_conv2d_9_bias_m_read_readvariableop2savev2_nadam_dense_16_kernel_m_read_readvariableop0savev2_nadam_dense_16_bias_m_read_readvariableop2savev2_nadam_dense_17_kernel_m_read_readvariableop0savev2_nadam_dense_17_bias_m_read_readvariableop2savev2_nadam_dense_18_kernel_m_read_readvariableop0savev2_nadam_dense_18_bias_m_read_readvariableop2savev2_nadam_dense_19_kernel_m_read_readvariableop0savev2_nadam_dense_19_bias_m_read_readvariableop2savev2_nadam_conv2d_8_kernel_v_read_readvariableop0savev2_nadam_conv2d_8_bias_v_read_readvariableop2savev2_nadam_conv2d_9_kernel_v_read_readvariableop0savev2_nadam_conv2d_9_bias_v_read_readvariableop2savev2_nadam_dense_16_kernel_v_read_readvariableop0savev2_nadam_dense_16_bias_v_read_readvariableop2savev2_nadam_dense_17_kernel_v_read_readvariableop0savev2_nadam_dense_17_bias_v_read_readvariableop2savev2_nadam_dense_18_kernel_v_read_readvariableop0savev2_nadam_dense_18_bias_v_read_readvariableop2savev2_nadam_dense_19_kernel_v_read_readvariableop0savev2_nadam_dense_19_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *I
dtypes?
=2;	2
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
?: : : :  : :
?d?:?:
??:?:
??:?:	?:: : : : : : : : : : :?:?:?:?:::::?:?:?:?: : :  : :
?d?:?:
??:?:
??:?:	?:: : :  : :
?d?:?:
??:?:
??:?:	?:: 2(
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
?d?:!
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
:?:%!

_output_shapes
:	?: 

_output_shapes
::
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
: :
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
: :!
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
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::!

_output_shapes	
:?:! 

_output_shapes	
:?:!!

_output_shapes	
:?:!"

_output_shapes	
:?:,#(
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
?d?:!(
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
:?:%-!

_output_shapes
:	?: .

_output_shapes
::,/(
&
_output_shapes
: : 0

_output_shapes
: :,1(
&
_output_shapes
:  : 2

_output_shapes
: :&3"
 
_output_shapes
:
?d?:!4

_output_shapes	
:?:&5"
 
_output_shapes
:
??:!6

_output_shapes	
:?:&7"
 
_output_shapes
:
??:!8

_output_shapes	
:?:%9!

_output_shapes
:	?: :

_output_shapes
::;

_output_shapes
: 
?
h
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_964401

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
h
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_963639

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
h
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_964283

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
?1
?
C__inference_model_4_layer_call_and_return_conditional_losses_963788
input_5
conv2d_8_963752
conv2d_8_963754
conv2d_9_963757
conv2d_9_963759
dense_16_963764
dense_16_963766
dense_17_963770
dense_17_963772
dense_18_963776
dense_18_963778
dense_19_963782
dense_19_963784
identity?? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_8_963752conv2d_8_963754*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_9634562"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_963757conv2d_9_963759*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_9634832"
 conv2d_9/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_9634352!
max_pooling2d_4/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????d* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_9635062
flatten_4/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_16_963764dense_16_963766*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_9635252"
 dense_16/StatefulPartitionedCall?
 alpha_dropout_12/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_9635702"
 alpha_dropout_12/PartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)alpha_dropout_12/PartitionedCall:output:0dense_17_963770dense_17_963772*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_9635942"
 dense_17/StatefulPartitionedCall?
 alpha_dropout_13/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_9636392"
 alpha_dropout_13/PartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)alpha_dropout_13/PartitionedCall:output:0dense_18_963776dense_18_963778*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_9636632"
 dense_18/StatefulPartitionedCall?
 alpha_dropout_14/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_9637082"
 alpha_dropout_14/PartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)alpha_dropout_14/PartitionedCall:output:0dense_19_963782dense_19_963784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_9637322"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_5
?	
?
(__inference_model_4_layer_call_fn_964154

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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_9638302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
?
h
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_963570

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
h
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_963708

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
(__inference_model_4_layer_call_fn_963857
input_5
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_9638302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????((::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_5
?
~
)__inference_dense_19_layer_call_fn_964431

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
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_9637322
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
?
j
1__inference_alpha_dropout_12_layer_call_fn_964288

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
 *:
config_proto*(

CPU

GPU2*0,1,2,3,4,5J 8? *U
fPRN
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_9635652
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
k
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_963703

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
D__inference_dense_18_layer_call_and_return_conditional_losses_964363

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
input_58
serving_default_input_5:0?????????((<
dense_190
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?~
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
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?z
_tf_keras_network?y{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 40, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["flatten_4", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_12", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_12", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["alpha_dropout_12", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_13", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_13", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["alpha_dropout_13", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_14", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_14", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["alpha_dropout_14", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["dense_19", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 40, 40, 2]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 40, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 40, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["flatten_4", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_12", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_12", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["alpha_dropout_12", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_13", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_13", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["alpha_dropout_13", 0, 0, {}]]]}, {"class_name": "AlphaDropout", "config": {"name": "alpha_dropout_14", "trainable": true, "dtype": "float32", "rate": 0.2}, "name": "alpha_dropout_14", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["alpha_dropout_14", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["dense_19", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "BinaryAccuracy", "config": {"name": "binary_accuracy", "dtype": "float32", "threshold": 0.5}}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}, {"class_name": "Precision", "config": {"name": "precision_4", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall_4", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "SensitivityAtSpecificity", "config": {"name": "sensitivity_at_specificity_4", "dtype": "float32", "num_thresholds": 200, "specificity": 0.99}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 40, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 40, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
?	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 40, 2]}}
?	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 40, 32]}}
?
trainable_variables
 	variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
#trainable_variables
$	variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12800}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12800]}}
?
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AlphaDropout", "name": "alpha_dropout_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "alpha_dropout_12", "trainable": true, "dtype": "float32", "rate": 0.2}}
?

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
7trainable_variables
8	variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AlphaDropout", "name": "alpha_dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "alpha_dropout_13", "trainable": true, "dtype": "float32", "rate": 0.2}}
?

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 200, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AlphaDropout", "name": "alpha_dropout_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "alpha_dropout_14", "trainable": true, "dtype": "float32", "rate": 0.2}}
?

Ekernel
Fbias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
Kiter

Lbeta_1

Mbeta_2
	Ndecay
Olearning_rate
Pmomentum_cachem?m?m?m?'m?(m?1m?2m?;m?<m?Em?Fm?v?v?v?v?'v?(v?1v?2v?;v?<v?Ev?Fv?"
	optimizer
v
0
1
2
3
'4
(5
16
27
;8
<9
E10
F11"
trackable_list_wrapper
v
0
1
2
3
'4
(5
16
27
;8
<9
E10
F11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

Qlayers
	variables
Rnon_trainable_variables
regularization_losses
Slayer_regularization_losses
Tmetrics
Ulayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):' 2conv2d_8/kernel
: 2conv2d_8/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

Vlayers
	variables
Wnon_trainable_variables
regularization_losses
Xlayer_regularization_losses
Ymetrics
Zlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_9/kernel
: 2conv2d_9/bias
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
trainable_variables

[layers
	variables
\non_trainable_variables
regularization_losses
]layer_regularization_losses
^metrics
_layer_metrics
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
trainable_variables

`layers
 	variables
anon_trainable_variables
!regularization_losses
blayer_regularization_losses
cmetrics
dlayer_metrics
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
#trainable_variables

elayers
$	variables
fnon_trainable_variables
%regularization_losses
glayer_regularization_losses
hmetrics
ilayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
?d?2dense_16/kernel
:?2dense_16/bias
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
)trainable_variables

jlayers
*	variables
knon_trainable_variables
+regularization_losses
llayer_regularization_losses
mmetrics
nlayer_metrics
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
-trainable_variables

olayers
.	variables
pnon_trainable_variables
/regularization_losses
qlayer_regularization_losses
rmetrics
slayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_17/kernel
:?2dense_17/bias
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
3trainable_variables

tlayers
4	variables
unon_trainable_variables
5regularization_losses
vlayer_regularization_losses
wmetrics
xlayer_metrics
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
7trainable_variables

ylayers
8	variables
znon_trainable_variables
9regularization_losses
{layer_regularization_losses
|metrics
}layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_18/kernel
:?2dense_18/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
=trainable_variables

~layers
>	variables
non_trainable_variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
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
Atrainable_variables
?layers
B	variables
?non_trainable_variables
Cregularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_19/kernel
:2dense_19/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Gtrainable_variables
?layers
H	variables
?non_trainable_variables
Iregularization_losses
 ?layer_regularization_losses
?metrics
?layer_metrics
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
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
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
_tf_keras_metric?{"class_name": "BinaryAccuracy", "name": "binary_accuracy", "dtype": "float32", "config": {"name": "binary_accuracy", "dtype": "float32", "threshold": 0.5}}
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
_tf_keras_metric?{"class_name": "Precision", "name": "precision_4", "dtype": "float32", "config": {"name": "precision_4", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Recall", "name": "recall_4", "dtype": "float32", "config": {"name": "recall_4", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?
?true_positives
?true_negatives
?false_positives
?false_negatives
?
thresholds
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "SensitivityAtSpecificity", "name": "sensitivity_at_specificity_4", "dtype": "float32", "config": {"name": "sensitivity_at_specificity_4", "dtype": "float32", "num_thresholds": 200, "specificity": 0.99}}
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
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Nadam/conv2d_8/kernel/m
!: 2Nadam/conv2d_8/bias/m
/:-  2Nadam/conv2d_9/kernel/m
!: 2Nadam/conv2d_9/bias/m
):'
?d?2Nadam/dense_16/kernel/m
": ?2Nadam/dense_16/bias/m
):'
??2Nadam/dense_17/kernel/m
": ?2Nadam/dense_17/bias/m
):'
??2Nadam/dense_18/kernel/m
": ?2Nadam/dense_18/bias/m
(:&	?2Nadam/dense_19/kernel/m
!:2Nadam/dense_19/bias/m
/:- 2Nadam/conv2d_8/kernel/v
!: 2Nadam/conv2d_8/bias/v
/:-  2Nadam/conv2d_9/kernel/v
!: 2Nadam/conv2d_9/bias/v
):'
?d?2Nadam/dense_16/kernel/v
": ?2Nadam/dense_16/bias/v
):'
??2Nadam/dense_17/kernel/v
": ?2Nadam/dense_17/bias/v
):'
??2Nadam/dense_18/kernel/v
": ?2Nadam/dense_18/bias/v
(:&	?2Nadam/dense_19/kernel/v
!:2Nadam/dense_19/bias/v
?2?
(__inference_model_4_layer_call_fn_963925
(__inference_model_4_layer_call_fn_964154
(__inference_model_4_layer_call_fn_964183
(__inference_model_4_layer_call_fn_963857?
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
!__inference__wrapped_model_963429?
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
input_5?????????((
?2?
C__inference_model_4_layer_call_and_return_conditional_losses_964125
C__inference_model_4_layer_call_and_return_conditional_losses_963788
C__inference_model_4_layer_call_and_return_conditional_losses_964073
C__inference_model_4_layer_call_and_return_conditional_losses_963749?
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
)__inference_conv2d_8_layer_call_fn_964203?
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
D__inference_conv2d_8_layer_call_and_return_conditional_losses_964194?
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
)__inference_conv2d_9_layer_call_fn_964223?
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
D__inference_conv2d_9_layer_call_and_return_conditional_losses_964214?
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
0__inference_max_pooling2d_4_layer_call_fn_963441?
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
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_963435?
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
*__inference_flatten_4_layer_call_fn_964234?
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_964229?
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
)__inference_dense_16_layer_call_fn_964254?
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
D__inference_dense_16_layer_call_and_return_conditional_losses_964245?
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
1__inference_alpha_dropout_12_layer_call_fn_964288
1__inference_alpha_dropout_12_layer_call_fn_964293?
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
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_964283
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_964278?
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
)__inference_dense_17_layer_call_fn_964313?
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
D__inference_dense_17_layer_call_and_return_conditional_losses_964304?
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
1__inference_alpha_dropout_13_layer_call_fn_964352
1__inference_alpha_dropout_13_layer_call_fn_964347?
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
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_964342
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_964337?
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
)__inference_dense_18_layer_call_fn_964372?
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
D__inference_dense_18_layer_call_and_return_conditional_losses_964363?
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
1__inference_alpha_dropout_14_layer_call_fn_964406
1__inference_alpha_dropout_14_layer_call_fn_964411?
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
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_964396
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_964401?
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
)__inference_dense_19_layer_call_fn_964431?
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
D__inference_dense_19_layer_call_and_return_conditional_losses_964422?
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
$__inference_signature_wrapper_963964input_5"?
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
!__inference__wrapped_model_963429}'(12;<EF8?5
.?+
)?&
input_5?????????((
? "3?0
.
dense_19"?
dense_19??????????
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_964278^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
L__inference_alpha_dropout_12_layer_call_and_return_conditional_losses_964283^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
1__inference_alpha_dropout_12_layer_call_fn_964288Q4?1
*?'
!?
inputs??????????
p
? "????????????
1__inference_alpha_dropout_12_layer_call_fn_964293Q4?1
*?'
!?
inputs??????????
p 
? "????????????
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_964337^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
L__inference_alpha_dropout_13_layer_call_and_return_conditional_losses_964342^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
1__inference_alpha_dropout_13_layer_call_fn_964347Q4?1
*?'
!?
inputs??????????
p
? "????????????
1__inference_alpha_dropout_13_layer_call_fn_964352Q4?1
*?'
!?
inputs??????????
p 
? "????????????
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_964396^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
L__inference_alpha_dropout_14_layer_call_and_return_conditional_losses_964401^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
1__inference_alpha_dropout_14_layer_call_fn_964406Q4?1
*?'
!?
inputs??????????
p
? "????????????
1__inference_alpha_dropout_14_layer_call_fn_964411Q4?1
*?'
!?
inputs??????????
p 
? "????????????
D__inference_conv2d_8_layer_call_and_return_conditional_losses_964194l7?4
-?*
(?%
inputs?????????((
? "-?*
#? 
0?????????(( 
? ?
)__inference_conv2d_8_layer_call_fn_964203_7?4
-?*
(?%
inputs?????????((
? " ??????????(( ?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_964214l7?4
-?*
(?%
inputs?????????(( 
? "-?*
#? 
0?????????(( 
? ?
)__inference_conv2d_9_layer_call_fn_964223_7?4
-?*
(?%
inputs?????????(( 
? " ??????????(( ?
D__inference_dense_16_layer_call_and_return_conditional_losses_964245^'(0?-
&?#
!?
inputs??????????d
? "&?#
?
0??????????
? ~
)__inference_dense_16_layer_call_fn_964254Q'(0?-
&?#
!?
inputs??????????d
? "????????????
D__inference_dense_17_layer_call_and_return_conditional_losses_964304^120?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_17_layer_call_fn_964313Q120?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_18_layer_call_and_return_conditional_losses_964363^;<0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_18_layer_call_fn_964372Q;<0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_19_layer_call_and_return_conditional_losses_964422]EF0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_19_layer_call_fn_964431PEF0?-
&?#
!?
inputs??????????
? "???????????
E__inference_flatten_4_layer_call_and_return_conditional_losses_964229a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????d
? ?
*__inference_flatten_4_layer_call_fn_964234T7?4
-?*
(?%
inputs????????? 
? "???????????d?
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_963435?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_4_layer_call_fn_963441?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_model_4_layer_call_and_return_conditional_losses_963749w'(12;<EF@?=
6?3
)?&
input_5?????????((
p

 
? "%?"
?
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_963788w'(12;<EF@?=
6?3
)?&
input_5?????????((
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_964073v'(12;<EF??<
5?2
(?%
inputs?????????((
p

 
? "%?"
?
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_964125v'(12;<EF??<
5?2
(?%
inputs?????????((
p 

 
? "%?"
?
0?????????
? ?
(__inference_model_4_layer_call_fn_963857j'(12;<EF@?=
6?3
)?&
input_5?????????((
p

 
? "???????????
(__inference_model_4_layer_call_fn_963925j'(12;<EF@?=
6?3
)?&
input_5?????????((
p 

 
? "???????????
(__inference_model_4_layer_call_fn_964154i'(12;<EF??<
5?2
(?%
inputs?????????((
p

 
? "???????????
(__inference_model_4_layer_call_fn_964183i'(12;<EF??<
5?2
(?%
inputs?????????((
p 

 
? "???????????
$__inference_signature_wrapper_963964?'(12;<EFC?@
? 
9?6
4
input_5)?&
input_5?????????(("3?0
.
dense_19"?
dense_19?????????