import time
from ops import *
from utils import *

class ResNet(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset

        if self.dataset_name == 'cifar10' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar100()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'fashion-mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_fashion()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'tiny' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiny()
            self.img_size = 64
            self.c_dim = 3
            self.label_dim = 200


        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.curr_epoch = 0
        self.batch_size = args.batch_size
        self.active_batch_size = args.active_batch_size
        self.iteration = self.active_batch_size // self.batch_size # len(self.train_x) // self.batch_size

        self.init_lr = args.lr

    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, curr_epoch=0, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))
            
            if self.curr_epoch > 0.6*self.epoch:
                learning_loss_0 = tf.stop_gradient(x)
            else:
                learning_loss_0 = x
            
            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            if self.curr_epoch > 0.6*self.epoch:
                learning_loss_1 = tf.stop_gradient(x)
            else:
                learning_loss_1 = x
            ########################################################################################################
            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))
            
            if self.curr_epoch > 0.6*self.epoch:
                learning_loss_2 = tf.stop_gradient(x)
            else:
                learning_loss_2 = x

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            if self.curr_epoch > 0.6*self.epoch:
                learning_loss_3 = tf.stop_gradient(x)
            else:
                learning_loss_3 = x

            ########################################################################################################


            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.label_dim, scope='logit')


            # learning loss module
            learning_loss_0 = global_avg_pooling(learning_loss_0)
            learning_loss_0 = fully_conneted(learning_loss_0, self.label_dim, scope='learning_loss_fc_0')
            learning_loss_0 = relu(learning_loss_0)

            learning_loss_1 = global_avg_pooling(learning_loss_1)
            learning_loss_1 = fully_conneted(learning_loss_1, self.label_dim, scope='learning_loss_fc_1')
            learning_loss_1 = relu(learning_loss_1)

            learning_loss_2 = global_avg_pooling(learning_loss_2)
            learning_loss_2 = fully_conneted(learning_loss_2, self.label_dim, scope='learning_loss_fc_2')
            learning_loss_2 = relu(learning_loss_2)
        
            learning_loss_3 = global_avg_pooling(learning_loss_3)
            learning_loss_3 = fully_conneted(learning_loss_3, self.label_dim, scope='learning_loss_fc_3')
            learning_loss_3 = relu(learning_loss_3)
           
            learning_loss = tf.concat([learning_loss_0,
                                                learning_loss_1,
                                                learning_loss_2,
                                                learning_loss_3], 
                                                axis=-1)
            learning_loss = fully_conneted(learning_loss, 1, scope='learning_loss_fc_4')
            learning_loss = relu(learning_loss)
            
            return x, learning_loss # learning loss is the predicted loss

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.test_inptus = tf.placeholder(tf.float32, [len(self.test_x), self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [len(self.test_y), self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits, self.train_predicted_loss = self.network(self.train_inptus)
        self.test_logits, self.test_predicted_loss = self.network(self.test_inptus, is_training=False, reuse=True)

        self.train_loss, self.train_accuracy = classification_loss(logit=self.train_logits, label=self.train_labels)
        self.loss_prediction_loss = loss_prediction_loss(logit=self.train_logits, label=self.train_labels, predicted_loss=self.train_predicted_loss, batch_size=self.batch_size)
        self.test_loss, self.test_accuracy = classification_loss(logit=self.test_logits, label=self.test_labels)
        
        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.train_loss += self.loss_prediction_loss
        self.test_loss += reg_loss


        """ Training """
        # self.optim = tf.contrib.opt.AdamWOptimizer(weight_decay=0.0005, learning_rate=self.lr).minimize(self.train_loss) # tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)
        self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)
        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)
        self.summary_train_loss_prediction_loss = tf.summary.scalar("loss_prediction_loss", self.loss_prediction_loss)
        
        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_loss_prediction_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.8) :
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
       
        # tr_idx = list(range(10000))
        # unannotated_idx_pool = list(range(10000, len(self.train_x)))
        # active_step = 1
        # for step in range(active_step):
        
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.3):
                epoch_lr = epoch_lr * 0.1
            elif epoch == int(self.epoch * 0.6):
                epoch_lr = epoch_lr * 0.1

            self.curr_epoch = epoch
            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }

                test_feed_dict = {
                    self.test_inptus : self.test_x,
                    self.test_labels : self.test_y
                }


                # update network
                _, summary_str, train_loss, train_accuracy = self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss, test_accuracy = self.sess.run(
                    [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)
        # save model for final step of each active_step
        # self.save(self.checkpoint_dir, counter)
    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_feed_dict = {
            self.test_inptus: self.test_x,
            self.test_labels: self.test_y
        }


        test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
        print("test_accuracy: {}".format(test_accuracy))
