import numpy as np 

def answer_question(question , model , rerankmodel , corpus_embed , corpus_list,llm_chain):
    # embeddings_1 = model.encode(question, batch_size=16, max_length=8192 ,)['dense_vecs']
    # embeddings_2 = corpus_embed
    # BGM3similarity = embeddings_1 @ embeddings_2.T

#==========================================================

    ALL_final_ans_list_ALL = []
    batch_size = 10 

    sentence_pairs = [[question, j] for j in corpus_list]

    listofscore = []
    compute_Score = range(0, len(sentence_pairs), batch_size)

    for i in compute_Score:
        batch_pairs = sentence_pairs[i:i+batch_size]
        allscore = model.compute_score(batch_pairs,
                                        max_passage_length=512,
                                        weights_for_different_modes=[0.4, 0.2, 0.4]) # sum: w[0]*dense_score + w[1]*sparse_score + w[2]*colbert_score
        listofscore.append(allscore) 

    score_ALL = []


    for score_dict in listofscore:

        score_ALL.extend(score_dict['colbert+sparse+dense'])

    ALL_final_ans_list_ALL.append(score_ALL)

#==========================================================

    topkindex = 15
    topk15scoresimilar_BGM3 = np.argsort(ALL_final_ans_list_ALL)[:,-topkindex:]
    # topk15scoresimilar_BGM3 = np.argsort(BGM3similarity)[-topkindex:]


    BGM3_1_retrieval = [corpus_list[i] for i in topk15scoresimilar_BGM3[0]]

    scores = []

    for passage in BGM3_1_retrieval:
        passage = str(passage)
        score = rerankmodel.compute_score([question, passage], normalize=True)
        scores.append(score)
        # print(passage[:20])

    highest_scoring_index = scores.index(max(scores))
    result_passage = BGM3_1_retrieval[highest_scoring_index]
    # print(f"Retrieval{result_passage[:20]}")
    # print(f"Question{question}")

    inputs = {"section": result_passage, "question": question}

    response = llm_chain.run(inputs)
    print(response)
    return response

