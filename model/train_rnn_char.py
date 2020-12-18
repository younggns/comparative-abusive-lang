# -*- coding: utf-8 -*-

"""
what    : train RNN model (char-level)
data    : twitter
"""
import os
import time
import argparse
import datetime

from layers.RNN_model_bi import *
from layers.RNN_process_data_char import *
from layers.RNN_evaluation import *
from layers.RNN_params import Params
from rnn_util import save_result


# for training
def train_step(sess, model, batch_gen):
    # raw_encoder_input_con, raw_encoder_seq_con, raw_encoder_type_con,
    # raw_encoder_input_ori, raw_encoder_seq_ori, raw_encoder_type_ori,
    # raw_label = batch_gen.get_batch(
    raw_encoder_input_con, raw_encoder_seq_con, raw_encoder_input_ori, raw_encoder_seq_ori, raw_label = batch_gen.get_batch(
        data=batch_gen.train_set, batch_size=model.batch_size, encoder_size=model.encoder_size, is_test=False)

    # prepare data which will be push from pc to placeholder
    input_feed = {}

    input_feed[model.encoder_inputs_c] = raw_encoder_input_con
    input_feed[model.encoder_seq_c] = raw_encoder_seq_con
    # input_feed[model.encoder_type_c] = raw_encoder_type_con

    input_feed[model.encoder_inputs_o] = raw_encoder_input_ori
    input_feed[model.encoder_seq_o] = raw_encoder_seq_ori
    # input_feed[model.encoder_type_o] = raw_encoder_type_ori

    input_feed[model.y_labels] = raw_label

    input_feed[model.dr_prob] = model.dr

    _, summary = sess.run([model.optimizer, model.summary_op], input_feed)

    return summary


def train_model(
        model,
        batch_gen,
        num_train_steps,
        valid_freq,
        is_save=0,
        graph_dir_name='default'):

    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    summary = None
    val_summary = None

    with tf.compat.v1.Session(config=config) as sess:

        writer = tf.compat.v1.summary.FileWriter(
            './graph/' + graph_dir_name, sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())

        early_stop_count = Params.MAX_EARLY_STOP_COUNT

        if model.use_glove == 1:
            sess.run(model.embedding_init, feed_dict={
                     model.embedding_placeholder: batch_gen.get_glove()})
            print('[completed] loading pre-trained embedding vector to placeholder')

        # if exists check point, starts from the check point
        ckpt = tf.train.get_checkpoint_state(
            os.path.dirname('save/' + graph_dir_name + '/'))
        if ckpt and ckpt.model_checkpoint_path:
            print('from check point!!!')
            saver.restore(sess, ckpt.model_checkpoint_path)

        initial_time = time.time()

        min_ce = 1000000
        best_dev_f1 = 0
        test_f1_at_best_dev = 0
        test_zip_at_best_dev = None

        for index in range(num_train_steps):

            try:
                # run train
                summary = train_step(sess, model, batch_gen)
                writer.add_summary(
                    summary, global_step=model.global_step.eval())

            except BaseException:
                print("excepetion occurs in train step")
                pass

            # run validation
            if (index + 1) % valid_freq == 0:

                dev_ce, dev_accr, dev_f1, dev_zip, dev_summary = run_test(
                    sess=sess, model=model, batch_gen=batch_gen, data=batch_gen.dev_set)

                writer.add_summary(
                    dev_summary, global_step=model.global_step.eval())

                end_time = time.time()

                if index > Params.CAL_ACCURACY_FROM:

                    if (dev_ce < min_ce):
                        min_ce = dev_ce

                        # save best result
                        if is_save is 1:
                            saver.save(sess, 'save/' + graph_dir_name +
                                       '/', model.global_step.eval())

                        early_stop_count = Params.MAX_EARLY_STOP_COUNT

                        test_ce, test_accr, test_f1, test_zip, _ = run_test(
                            sess=sess, model=model, batch_gen=batch_gen, data=batch_gen.test_set)

                        best_dev_f1 = dev_f1
                        test_f1_at_best_dev = test_f1
                        test_zip_at_best_dev = test_zip

                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print("early stopped")
                            break

                        test_f1 = 0
                        early_stop_count = early_stop_count - 1

                    print(str(int((end_time -
                                   initial_time) /
                                  60)) +
                          " mins" +
                          " step/seen/itr: " +
                          str(model.global_step.eval()) +
                          "/ " +
                          str(model.global_step.eval() *
                              model.batch_size) +
                          "/" +
                          str(round(model.global_step.eval() *
                                    model.batch_size /
                                    float(len(batch_gen.train_set)), 2)) +
                          "\tdev: " +
                          '{:.3f}'.format(dev_f1) +
                          "  test: " +
                          '{:.3f}'.format(test_f1) +
                          "  loss: " +
                          '{:.2f}'.format(dev_ce))

        writer.close()

        print('Total steps : {}'.format(model.global_step.eval()))

        # unzip final result
        accr_class, recall_class, f1_class, accr_avg, recall_avg, f1_avg = test_zip_at_best_dev

        print(accr_class, recall_class, f1_class)

        # result logging to file
        save_result(
            batch_gen,
            graph_dir_name,
            accr_class,
            recall_class,
            f1_class,
            accr_avg,
            recall_avg,
            f1_avg
        )


def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


def main(data_path, batch_size, encoder_size, num_layer, hidden_dim,
         num_train_steps, lr, is_save, graph_dir_name,
         use_glove,
         dr,
         # o_type, c_text, c_type,
         c_text,
         attn, ltc
         ):

    if is_save is 1:
        create_dir('save/')
        create_dir('save/' + graph_dir_name)

    create_dir('graph/')
    create_dir('graph/' + graph_dir_name)

    batch_gen = ProcessData(data_path)

    model = SingleEncoderModelBi(
        dic_size=batch_gen.dic_size,
        use_glove=use_glove,
        batch_size=batch_size,
        encoder_size=encoder_size,
        num_layer=num_layer,
        lr=lr,
        hidden_dim=hidden_dim,
        dr=dr,
        # o_type = o_type,
        # c_text = c_text,
        # c_type = c_type,
        attn=attn,
        ltc=ltc
    )

    model.build_graph()

    valid_freq = int(len(batch_gen.train_set) *
                     Params.EPOCH_PER_VALID_FREQ / float(batch_size)) + 1
    print("[Info] Valid Freq = " + str(valid_freq))

    train_model(model, batch_gen, num_train_steps,
                valid_freq, is_save, graph_dir_name)


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--encoder_size', type=int, default=750)

    p.add_argument('--num_layer', type=int, default=1)
    p.add_argument('--hidden_dim', type=int, default=50)

    p.add_argument('--num_train_steps', type=int, default=10000)
    p.add_argument('--lr', type=float, default=1e-1)
    p.add_argument('--is_save', type=int, default=0)
    p.add_argument('--graph_prefix', type=str, default="default")

    p.add_argument('--use_glove', type=int, default=0)
    p.add_argument('--dr', type=float, default=1.0)

    # p.add_argument('--o_type', type=int, default=0)
    p.add_argument('--c_text', type=int, default=0)
    # p.add_argument('--c_type', type=int, default=0)

    p.add_argument('--attn', type=int, default=0)
    p.add_argument('--ltc', type=int, default=0)

    args = p.parse_args()

    embed_train = ''
    if not Params.EMBEDDING_TRAIN:
        embed_train = 'F'

    # graph_name = args.graph_prefix + '_b' + str(args.batch_size) + '_es' + str(args.encoder_size) + '_L' + str(args.num_layer) + '_H' + str(args.hidden_dim) + '_G' + str(args.use_glove) + embed_train + '_dr' + str(args.dr) + '_o_type' + str(args.o_type) + '_c_text' + str(args.c_text) + '_c_type' + str(args.c_type) + '_attn' + str(args.attn) + '_ltc' + str(args.ltc) + '_D' + (args.data_path).split('/')[-2]

    graph_name = args.graph_prefix + \
        '_b' + str(args.batch_size) + \
        '_es' + str(args.encoder_size) + \
        '_L' + str(args.num_layer) + \
        '_H' + str(args.hidden_dim) + \
        '_G' + str(args.use_glove) + embed_train + \
        '_dr' + str(args.dr) + \
        '_c_text' + str(args.c_text) + \
        '_attn' + str(args.attn) + \
        '_ltc' + str(args.ltc) + \
        '_D' + (args.data_path).split('/')[-2]

    graph_name = graph_name + '_' + datetime.datetime.now().strftime("%m-%d-%H-%M")

    main(
        data_path=args.data_path,
        batch_size=args.batch_size,
        encoder_size=args.encoder_size,
        num_layer=args.num_layer,
        hidden_dim=args.hidden_dim,
        num_train_steps=args.num_train_steps,
        lr=args.lr,
        is_save=args.is_save,
        graph_dir_name=graph_name,
        use_glove=args.use_glove,
        dr=args.dr,
        # o_type=args.o_type,
        c_text=args.c_text,
        # c_type=args.c_type,
        attn=args.attn,
        ltc=args.ltc
    )
