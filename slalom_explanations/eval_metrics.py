#@profile
def compute_correlations(config, trainer, model_trained, tokenizer,
                importances, use_dataset, max_seq_len):
    """ Compute correlation scores wrt the ground truth """
    # dataloader_test = torch.utils.data.DataLoader(trainer.ds_tokenized['test'], batch_size=config.batch_size, shuffle=False)
    # trainer.test(dataloader_test, epoch_num= 999, log_attn = True)
    corrs = {}
    corrs["last_acc"] = trainer.last_acc
    for mygt in config.ground_truth:
        corrs[mygt]= {"raw":[], "rollout":[]}
    corrs["sparsity"] = []

    if len(config.ground_truth) == 0:
        return corrs

    special_tokens = tokenizer.all_special_tokens
    # special_tokens = None
    print(f'compute correlation without special tokens: {special_tokens}')
    # compute correlation
    print("computing correlation")
    for i in tqdm(range(use_dataset['test'].shape[0]), total=use_dataset['test'].shape[0]):
        # get decoded tokens
        input_ids = tokenizer.encode(use_dataset['test'][i]['text'])[:max_seq_len]
        decoded_tokens = []
        for input in input_ids:
            decoded_tokens.append(tokenizer.convert_ids_to_tokens([input])[0])

        # get the attention matrix
        input_ids_tensor = torch.tensor([input_ids]).to(config.device)
        model_output = model_trained(input_ids_tensor)
        attentions = model_output['attentions']
        attentions = [att.detach().cpu() for att in attentions]
        attention_mat = torch.stack(attentions) # Remove empty second dim
        # sample_nr = 'sample_' + str(i).zfill(5)
        # eps = 'epoch_' + str(999).zfill(3)
        # p = os.path.join(trainer.logger_att_mat.path, trainer.logger_att_mat.run_name, eps, sample_nr)
        # if not os.path.exists(p):
        #     os.makedirs(p)
        # np.save(os.path.join(p, 'att_mat_2.npy'), attention_mat)
        # compute attention representations (Raw and Rollout)
        #print(attention_mat.shape)
        att_models = AttentionModels(attention_mat=attention_mat)
        att_raw = att_models.get_raw_attention()[:,0] # batch index 0
        att_rollout = att_models.get_attention_rollout()[:,0] # batch index 0
        #sparsity = att_models.compute_head_wise_sparsity(start_token=0)
        #corrs["sparsity"].append(sparsity.tolist())
        #print(np.sum(att_raw))
        # compute correlation
        for key, gtimp in importances.items():
            # GPT2 uses last token for prediction, so set start token to last token
            if config.model == 'distilbert':
                start_token = 0
            elif config.model == 'gpt2':
                start_token = len(decoded_tokens) - 1
            correlation_raw = Correlation(att_matrix=att_raw.numpy(), decoded_tokens=decoded_tokens, importances=gtimp, special_tokens=special_tokens)
            correlation_rollout = Correlation(att_matrix=att_rollout.numpy(), decoded_tokens=decoded_tokens, importances=gtimp, special_tokens=special_tokens)
            corrs[key]["raw"].append(correlation_raw.compute_spearmanr(layer=-1, start_token=start_token))  # layer=-1 means we average over all layers
            corrs[key]["rollout"].append(correlation_rollout.compute_spearmanr(layer=-1, start_token=start_token))
    return corrs



if config.evalevery:
        corrs = []
        step_cnt = 0
        for ep in range(config.epochs):
            print(step_cnt)
            step_cnt = trainer.train(epochs=1, test_interval=1, batch_size=config.batch_size, lr=5e-5, 
                    logsteps = config.log_attn_steps, start_steps = step_cnt, start_epoch=ep)
            model_trained = trainer.get_model()
            if ref_importances is None:
                ref_importances = compute_ref_importances(config, model_trained, dataset, tokenizer, max_seq_len)
            corrs.append(compute_correlations(config, trainer, model_trained, tokenizer, ref_importances, dataset, max_seq_len))
            # print(corrs)
            #logger.update_result(corrs, model=config.model, layers=config.layers, heads=config.n_heads)
    else:
        trainer.train(epochs=config.epochs, test_interval=1, batch_size=config.batch_size, lr=1e-4, logsteps = config.log_attn_steps)
        model_trained = trainer.get_model()
        ref_importances = compute_ref_importances(config, model_trained, dataset, tokenizer, max_seq_len)
        corrs = compute_correlations(config, trainer, model_trained, tokenizer, ref_importances, dataset, max_seq_len)


