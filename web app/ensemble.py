import tensorflow as tf
def ensemble(models, model_input):
        outputs = [model.outputs[0] for model in models]
        y =tf.keras.layers.Average()(outputs)
        model =tf.keras.Model(model_input,y,name='ensemble')
        return model