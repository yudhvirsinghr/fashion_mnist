from core.model import CNN_model
from core.load_data import train_data, train_label, test_data, test_label
from config import EPOCHS, loss, optimizer, batch_size, save_path

model = CNN_model()
model.build(input_shape=(None , 28, 28, 1))
print(model.model().summary())

model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['accuracy'])

model.fit(train_data, train_label,
        batch_size = batch_size,
        epochs=EPOCHS,
        validation_data=(test_data, test_label))

model.save_weights(save_path + 'mymodel')
