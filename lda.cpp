#include <vector>
#include <iostream>
#include <algorithm>
extern "C" {
    void lda(unsigned long * X, unsigned long M, unsigned long N, unsigned long K, float * doc_topic, float * topic_word, unsigned long n_iter, float alpha, float beta);
}

void lda(unsigned long * X, // matrix of all corpus, X[m, n]: count of word n in doc m
         unsigned long M, // total doc count
         unsigned long N, // total word count
         unsigned long K, // topic number
         float * doc_topic, // return doc_topic matrix, M x K
         float * topic_word, // return topic_word matrix, K x N
         unsigned long n_iter = 100, // iter times
         float alpha = 0.1, // Dirichlet parameter for distribution over topics
         float beta = 0.01 // Dirichlet parameter for distribution over words
         ){
    std::cout<<"running LDA..."<<std::endl;
    std::vector<float> randoms(10000);
    for(unsigned long i; i < randoms.size(); ++i) randoms[i] = 1.0 * i / randoms.size();

    std::vector<unsigned long> nzw_(K * N, 0); // Matrix of counts recording topic-word assignments in final iteration. shape = [n_topics, n_words]
    std::vector<unsigned long> ndz_(M * K, 0); // Matrix of counts recording document-topic assignments in final iteration. shape = [n_doc, n_topics]
    std::vector<unsigned long> nz_(K, 0);
    std::vector<float> alphas(K, alpha);
    std::vector<float> betas(N, beta);
    float beta_sum = N * beta;
    std::vector<unsigned long> WS; // all words in a vector
    std::vector<unsigned long> DS; // docid for each element in WS
    std::vector<unsigned long> ZS; // topic for each element in WS
    std::cout<<"initialization..."<<std::endl;
    // initialization
    for(unsigned long i = 0; i < M * N; ++i){
        if (X[i] == 0) continue;
        WS.insert(WS.end(), X[i], i % N);
        DS.insert(DS.end(), X[i], i / N);
        for(unsigned j = 0; j < X[i]; ++j){
            unsigned long zj = rand() % K;
            ZS.push_back(zj);
            ++nzw_[zj * N + i % N];
            ++ndz_[i / N * K + zj];
            ++nz_[zj];
        }
    }
    std::cout<<"Gibbs Sampling..."<<std::endl;
    // Gibbs Sampling
    for(unsigned iter_i = 0; iter_i < n_iter; ++iter_i){
        std::cout<<"iter: "<<iter_i + 1<<"\r";
        for(unsigned long i = 0; i < WS.size(); ++i){
            --nzw_[ZS[i] * N + WS[i]];
            --ndz_[DS[i] * K + ZS[i]];
            --nz_[ZS[i]];
            std::vector<float> dist_sum(K, 0);
            float dist_cum = 0;
            for(unsigned long k = 0; k < K; ++k){
                dist_cum += (nzw_[k * N + WS[i]] + betas[WS[i]]) / (nz_[k] + beta_sum) * (ndz_[DS[i] * K + k] + alphas[k]);
                dist_sum[k] = dist_cum;
            }
            unsigned long z_new = std::lower_bound(dist_sum.begin(), dist_sum.end(), randoms[rand() % randoms.size()] * dist_cum) - dist_sum.begin();
            ++nzw_[z_new * N + WS[i]];
            ++ndz_[DS[i] * K + z_new];
            ++nz_[z_new];
            ZS[i] = z_new;
        }
    }
    std::cout<<"Finish Iter"<<std::endl;

    std::vector<float> topic_sum(K, 0);
    for(unsigned long i = 0; i < K * N; ++i){
        topic_word[i] = nzw_[i] + betas[i % N];
        topic_sum[i / N] += topic_word[i];
    }
    for(unsigned long i = 0; i < K * N; ++i){
        topic_word[i] /= topic_sum[i / N];
        }
    std::vector<float> doc_sum(M, 0);
    for(unsigned long i = 0; i < M * K; ++i){
        doc_topic[i] = ndz_[i] + alphas[i % K];
        doc_sum[i / K] += doc_topic[i];
    }
    for(unsigned long i = 0; i < M * K; ++i){
        doc_topic[i] /= doc_sum[i / K];
    }
}

int main(){
    unsigned long X[3 * 4] = {1, 2, 3, 0, 3, 6, 7, 1, 2, 4, 6, 4};
    int k = 2;
    float doc_topic[3 * k];
    float topic_word[k * 4];
    lda(X, 3, 4, k, doc_topic, topic_word);
}
