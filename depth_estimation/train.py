import os, sys, glob, time, pathlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Kerasa / TensorFlow
from loss import depth_loss_function
from utils import predict, save_images, load_test_data
from model import create_model
from model_resnet import create_model_resnet
from data import get_nyu_train_test_data, get_unreal_train_test_data
from callbacks import get_nyu_callbacks

from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--data', default='nyu', type=str, help='Training dataset.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=10.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='densedepth_nyu', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')
parser.add_argument('--dnetVersion', type = str, default= 'medium' , help='Choice of densenet from small, medium or large.')
parser.add_argument('--resnet50', dest='resnet50', action='store_true', help='Train a Resnet 50 model.')

args = parser.parse_args()

# Inform about multi-gpu training
if args.gpus == 1: 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('Will use GPU ' + args.gpuids)
else:
    print('Will use ' + str(args.gpus) + ' GPUs.')

# Create the model
if args.resnet50:  # if want a resnet model
    model = create_model_resnet(existing=args.checkpoint)
else: # choose densetnet model
    if args.dnetVersion == 'small':
        model = create_model(existing=args.checkpoint, is121 = True)
    if args.dnetVersion == 'medium':
        model = create_model( existing=args.checkpoint )
    if args.dnetVersion == 'large':
        model = create_model( existing=args.checkpoint, is_twohundred=True )

# Data loaders
if args.data == 'nyu': train_generator, test_generator = get_nyu_train_test_data( args.bs )
if args.data == 'unreal': train_generator, test_generator = get_unreal_train_test_data( args.bs )

# Training session details
runID = str(int(time.time())) + '-n' + str(len(train_generator)) + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
outputPath = './models/'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

 # (optional steps)
if True:
    # Keep a copy of this training script and calling arguments
    with open(__file__, 'r') as training_script: training_script_content = training_script.read()
    training_script_content = '#' + str(sys.argv) + '\n' + training_script_content
    with open(runPath+'/'+__file__, 'w') as training_script: training_script.write(training_script_content)

    # Generate model plot
    plot_model(model, to_file=runPath+'/model_plot.svg', show_shapes=True, show_layer_names=True)

    # Save model summary to file
    from contextlib import redirect_stdout
    with open(runPath+'/model_summary.txt', 'w') as f:
        with redirect_stdout(f): model.summary()

# Multi-gpu setup:
basemodel = model
if args.gpus > 1: model = multi_gpu_model(model, gpus=args.gpus)

# Optimizer
optimizer = Adam(lr=args.lr, amsgrad=True)

# Compile the model
print('\n\n\n', 'Compiling model..', runID, '\n\n\tGPU ' + (str(args.gpus)+' gpus' if args.gpus > 1 else args.gpuids)
        + '\t\tBatch size [ ' + str(args.bs) + ' ] ' + ' \n\n')
model.compile(loss=depth_loss_function, optimizer=optimizer)

print('Ready for training!\n')

# Callbacks
callbacks = []
if args.data == 'nyu': callbacks = get_nyu_callbacks(model, basemodel, train_generator, test_generator, load_test_data() if args.full else None , runPath)
if args.data == 'unreal': callbacks = get_nyu_callbacks(model, basemodel, train_generator, test_generator, load_test_data() if args.full else None , runPath)

# Start training
model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=args.epochs, shuffle=True)

# Save the final trained model:
basemodel.save(runPath + '/model.h5')
