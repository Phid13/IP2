import matplotlib.pyplot as plt
from model import *
from data import *
from labels import *
from skimage import measure

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = "data/RednBlue/Axial/Label"
gen_labels(path)

# change

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model.fit(myGene,steps_per_epoch=300,epochs=1)
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict(testGene,30,verbose=1)
saveResult("data/membrane/results",results)

# fig, ax = plt.subplots()
# flag_multi_class = False
# for i,item in enumerate(results):
#     image = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
#     r = image

#     ax.imshow(r, cmap=plt.cm.gray)
#     # Find contours at a constant value of 0.8
#     contours = measure.find_contours(r, 0.8)

#     for contour in contours:
#         ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
#     ax.axis('image')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.savefig("data/membrane/test/image_%d"%i)
