import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import (
    ModelCheckpoint, TensorBoard, LearningRateScheduler
)
from tensorflow.python.keras.models import load_model


custom_objects = {'bce_dice_loss': bce_dice_loss, 'dice_coeff': dice_coeff}


def fix_tf_rtx_gpu_bug():
    """Run to set up TensorFlow session with RTX series NVidia GPUs
    
    Compensates for cuDNN bug in tensorflow + RTX series NVidia GPUs.
    """
    if tf.version.VERSION.startswith('1'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf_session = tf.Session(config=config)
        tf.keras.backend.set_session(tf_session)
    elif tf.version.VERSION.startswith('2'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        else:
            raise Exception('Unsupported version of tensorflow encountered ({})'.format(
                tf.version.VERSION))


def save_init(model, model_name):
    filename = log_dir / '{}_init_weights.h5'.format(model_name)
    if filename.exists():
        print('Initial weights already saved.')
    else:
        print('Saving initial weights...')
        model.save_weights(str(filename))


def make_callbacks(subdir, schedule=[(1e-3, 400)]):
    outdir = log_dir / subdir
    weights_file = outdir / 'weights.hdf5'
    assert not weights_file.exists(), 'Trained weights already exist for this subdir'
    
    return [
        ModelCheckpoint(filepath=str(weights_file), monitor='val_loss',
                        save_best_only=True, verbose=1),
        TensorBoard(log_dir=str(outdir / 'logs')),
        LearningRateScheduler(lambda epoch: schedule_steps(epoch, schedule))
    ]


def save_history(subdir, history):
    with open(log_dir / subdir / 'history.pkl','wb') as f:
        pickle.dump({'history': history.history, 'epoch': history.epoch}, f)


def train_model(model, subdir, model_name=None, epochs=400, schedule=None, check=False):
    # First check output names match current flattener names
    assert(all([m == f for m, f in zip(model.output_names, flattener.names())]))
    
    if schedule is None:
        schedule = [(1e-3, epochs)]
    if model_name is None:
        model_name = subdir
    init_weights_file = log_dir / '{}_init_weights.h5'.format(model_name)
    assert init_weights_file.exists(), 'Initial weights have not been saved for this model'
    
    model.load_weights(str(init_weights_file))
    
    if check:
        return
    
    history = model.fit_generator(
        generator=train_gen, validation_data=val_gen, epochs=400,
        callbacks=make_callbacks(subdir))
    save_history(subdir, history)


def load_history(subdir):
    with open(log_dir / subdir / 'history.pkl','rb') as f:
        return pickle.load(f)


def get_best_and_worst(model, gen):
    best = {}
    worst = {}
    for i in range(len(gen)):
        inputim, target = gen[i]
        pred = model.predict(inputim)
        for l, output in enumerate(model.output_names):
            if output not in best:
                best[output] = []
            if output not in worst:
                worst[output] = []
            target_bin = target[l][...,0] > 0
            pred_bin = np.squeeze(pred[l]) > 0.5
            for p, t, im in zip(pred_bin, target_bin, inputim):
                u = np.sum(p | t)
                if u == 0:
                    # Ignore any empty traps
                    continue
                iou = np.sum(p & t) / u
                out = (iou, i, p, t, im)
                if len(best[output])<8:
                    best[output].append(out)
                    worst[output].append(out)
                else:
                    best_IoUs = [b[0] for b in best[output]]
                    best_minind = np.argmin(best_IoUs)
                    if iou > best_IoUs[best_minind]:
                        best[output][best_minind] = out
                    worst_IoUs = [w[0] for w in worst[output]]
                    worst_maxind = np.argmax(worst_IoUs)
                    if iou < worst_IoUs[worst_maxind]:
                        worst[output][worst_maxind] = out
        
    return best, worst
