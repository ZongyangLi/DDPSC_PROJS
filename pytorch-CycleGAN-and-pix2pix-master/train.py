import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
opt.save_epoch_freq = 2
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch, total_steps)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()


# python train.py --dataroot /media/zli/data/dcgan/terra_s2_rgb/shink_crop/ --name terra_gray_to_color --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm instance --niter 2 --niter_decay 12 --batchSize 16 --gpu_ids 0,1 --checkpoints_dir /media/zli/data/check_point/

# python train.py --dataroot /media/zli/data/dcgan/terra_s2_rgb/shink_crop/ --name terra_rgb_crop --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --niter 2 --niter_decay 10

# python train.py --dataroot /media/zli/data/dcgan/700 --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch
# --use_dropout option, or --which_model_netD=n_layers with --n_layers_D=4

#python train.py --dataroot /media/zli/data/dcgan/s6_hard_green/ganNewImages/ --name color_to_color_0.2 --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 70 --dataset_mode aligned --no_lsgan --norm instance --niter 2 --niter_decay 8 --batchSize 8 --gpu_ids 0,1 --checkpoints_dir /media/zli/data/check_point