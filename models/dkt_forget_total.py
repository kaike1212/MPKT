import torch

from torch import softmax
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, ReLU, init

device = "cpu" if not torch.cuda.is_available() else "cuda"


class DKTForgetTotal(Module):
    def __init__(self, num_question, num_skill, num_rgap, num_pcount, num_acount, forget_window, emb_dim, dropout=0.2,
                 emb_type='qid', emb_path=""):
        super().__init__()
        self.model_name = "dkt_forget_total"
        self.num_question = num_question
        self.num_skill = num_skill
        self.emb_dim = emb_dim
        self.forget_window = forget_window
        self.hidden_size = emb_dim
        self.emb_type = emb_type

        # self.interaction_emb = Embedding(self.num_question * 2, self.emb_dim)

        self.emb_table_skill = Embedding(self.num_skill, self.emb_dim)
        self.emb_table_question = Embedding(self.num_question, self.emb_dim)
        self.emb_table_response = Embedding(2, emb_dim)
        # self.c_integration = CIntegration(num_rgap, num_sgap, num_pcount, emb_size)
        # ntotal = num_rgap + num_sgap + num_pcount*2
        self.input_lstm_layer = LSTM(self.emb_dim, self.hidden_size)
        self.dropout_layer = Dropout(dropout)
        self.forget_lstm_layer = LSTM(self.emb_dim, self.hidden_size)

        self.linear1 = Linear(self.emb_dim * 2, emb_dim)
        self.linear2 = Linear(self.emb_dim * 3, emb_dim)
        self.linear3 = Linear(self.emb_dim * 2, emb_dim)
        self.fc1 = Linear(self.emb_dim * 3, emb_dim)
        self.fc2 = Linear(self.emb_dim * 3, emb_dim)
        self.fc3 = Linear(self.emb_dim * 2, 1)

        self.fc4 = Linear(self.emb_dim * 2, emb_dim)
        self.fc5 = Linear(self.emb_dim * 2, emb_dim)

        self.conceptIntegration = ConceptIntergation(num_skill, self.emb_table_skill)
        self.forgetIntegration = ForgetIntegration(self.conceptIntegration, forget_window, num_rgap,
                                                   num_pcount, num_acount, emb_dim, dropout)

        # 随机初始化
        init.xavier_uniform_(self.emb_table_skill.weight.data)
        init.xavier_uniform_(self.emb_table_question.weight.data)
        init.xavier_uniform_(self.emb_table_response.weight.data)

        # 初始化 LSTM 的权重
        for name, param in self.input_lstm_layer.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param.data)
        for name, param in self.forget_lstm_layer.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param.data)

        # 初始化 Linear 层的权重
        init.xavier_uniform_(self.linear1.weight.data)
        init.xavier_uniform_(self.linear2.weight.data)
        init.xavier_uniform_(self.linear3.weight.data)
        init.xavier_uniform_(self.fc1.weight.data)
        init.xavier_uniform_(self.fc2.weight.data)
        init.xavier_uniform_(self.fc3.weight.data)
        init.xavier_uniform_(self.fc4.weight.data)
        init.xavier_uniform_(self.fc5.weight.data)

    # q 问题 [batch,199]
    # r 回答 [batch,199]
    # dgaps 间隔 { rgaps  sgaps   pcounts [batch,199]
    def forward(self, q, c, shift_q, shift_c, r, dgaps):
        q, r = q.to(device), r.to(device)
        c = c.to(device)
        shift_q, shift_c = shift_q.to(device), shift_c.to(device)
        batch,lenn=q.shape
        # x = (q + num_question * r).to(device)
        emb_q = self.emb_table_question(q)
        emb_r = self.emb_table_response(r)

        # 1.1只有题目和回答的输入特征
        '''
        x = torch.cat((emb_q, emb_r), -1)
        # xemb = self.interaction_emb(x)  # xemb [batch,199,200]
        xemb = self.linear1(x)
        '''
        # x = torch.cat((emb_q, emb_c, emb_r), -1)
        # 1.2整合技能
        concept = self.conceptIntegration(c)  # [batch,len,num_skill,emb_dim]
        # 1.2.2 取注意力机制
        concept = self.attention(emb_q, concept)
        x = torch.cat((emb_q, emb_r, concept), -1)
        xemb = self.linear2(x)

        h, _ = self.input_lstm_layer(xemb)
        input_h = self.dropout_layer(h) #[batch,len,num_skill,dim]

        # theta_in = self.forgetIntegration(xemb, c, dgaps["rgaps"], dgaps["sgaps"], dgaps["pcounts"],
        #                                  dgaps["acounts"])
        # theta_in = self.c_integration(xemb, dgaps["rgaps"].to(device).long(), dgaps["sgaps"].to(device).long(),
        #                              dgaps["pcounts"].to(device).long())
        # 2.1 双序列
        forget_h = self.forgetIntegration(c, dgaps["rgaps"], dgaps["pcounts"],
                                          dgaps["acounts"])  # [batch,len,dim]
        # 4.LSTM
        forget_h, _ = self.forget_lstm_layer(forget_h)
        forget_h = self.dropout_layer(forget_h)

        # 3.预测模块
        # 3.1 input 问题+概念

        shift = self.emb_table_question(shift_q)  # [batch,len,dim]
        # shift 刚开始就是有题目序号 后续加入遗忘特征 再加入概念特征 再加入题面信息
        # 复杂一点，加入概念特征的shift
        shift_concept = self.conceptIntegration(shift_c)
        shift_concept = self.attention(shift, shift_concept)
        shift = torch.cat((shift, shift_concept), -1)
        shift = self.linear3(shift)

        # 3.2.1 加入遗忘特征、概念特征的shift
        shift_forget = self.forgetIntegration(shift_c, dgaps["shft_rgaps"], dgaps["shft_pcounts"],
                                              dgaps["shft_acounts"])  # [batch,len,dim]
        '''
        theta = torch.mul(shift, shift_forget)
        shift_forget = torch.cat((theta, shift), -1)  # [batch,len,2*dim]
        x1 = torch.relu(self.fc1(torch.cat((input_h, shift_forget), -1)))
        x2 = torch.relu(self.fc2(torch.cat((shift_forget, forget_h), -1)))
        '''
        # 3.2.2 分开预测

        # 3.2.2 分开预测    [batch,len,num_skill,dim]   [batch,len,dim*2]
        # knowledge_state=self.knowledge_linear(torch.cat((input_h,forget_h),-1)).view(batch, lenn, self.num_skill, self.emb_dim)
        # #[batch,len,num_skill] #[batch,len,num_skill,dim]
        # concept_vector = self.conceptIntegration.getConcept(shift_c).unsqueeze(2)
        # pre_y=torch.matmul(concept_vector,knowledge_state).squeeze(2) #[batch,len,dim]
        # y = torch.sigmoid(self.fc3(torch.cat((pre_y,shift,shift_forget),-1)))

        x1 = torch.relu(self.fc4(torch.cat((shift, input_h), -1)))
        # x2 = torch.relu(self.fc5(torch.cat((forget_h, shift_forget), -1)))
        x2 = torch.relu(self.fc5(torch.cat((shift_forget, forget_h), -1)))

        y = self.fc3(torch.cat((x1, x2), -1))
        y = torch.sigmoid(y)
        y = y.squeeze(dim=-1)
        return y

    def attention(self, q, c):
        # q [batch,len,dim]
        # c [batch,len,num_skill,dim]
        scores = torch.matmul(q.unsqueeze(2), c.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attention_weights = softmax(scores, -1)  # [batch, len, 1, num_skill]
        # Apply attention weights to value
        weighted_sum = torch.matmul(attention_weights, c).squeeze(2)  # [batch, len, input_dim]
        return weighted_sum

# 得到 C_t
class ConceptIntergation(Module):
    def __init__(self, num_skill, emb_table_skill):
        super(ConceptIntergation, self).__init__()
        self.emb_table_skill = emb_table_skill
        self.concepts_eye = torch.eye(num_skill + 1)
        self.sequence_tensor = torch.arange(num_skill)
        self.num_skill = num_skill

    def forward(self, concepts):
        concepts = concepts.to(device)
        concept_vector1= self.getConcept(concepts)
        concept_vector1 = concept_vector1.unsqueeze(-1)  # [batch,len,37,1]
        concept_vector2 = self.emb_table_skill(self.sequence_tensor.to(device))
        concept_vector2 = concept_vector2.unsqueeze(0).unsqueeze(0).to(device)  # vector2  #[1,1,37,dim]
        concept_true = concept_vector1 * concept_vector2  # [batch,len,num_skill,emb_dim]
        return concept_true
    def getConcept(self,concepts):
        concepts = concepts.to(device)
        self.concepts_eye = self.concepts_eye.to(device)
        shape_concept = self.concepts_eye[concepts][:, :, :, :-1]  # [37,37] #[batch,len,9,37]
        concept_vector1 = torch.sum(shape_concept, dim=-2).to(device)  # [batch,len,37]
        return concept_vector1


# 得到 C_t + forget_t
class ForgetIntegration(Module):
    def __init__(self, conceptIntegration, forget_window, num_rgap, num_pcount, num_acount, emb_dim, dropout):
        super(ForgetIntegration, self).__init__()
        self.emb_dim = emb_dim
        self.num_skill = conceptIntegration.num_skill
        self.emb_table_skill = conceptIntegration.emb_table_skill
        self.forget_window = forget_window
        self.forget_len = len(forget_window) + 1

        self.rgap_eye = torch.eye(num_rgap+5)
        self.pcount_eye = torch.eye(num_pcount+5)
        self.acount_eye = torch.eye(num_acount+5)

        self.ntotal = num_rgap + num_pcount + num_acount + 15
        self.ntotal = (len(forget_window) + 1) * self.ntotal

        self.cemb = Linear(self.ntotal, emb_dim, bias=False)

        self.lstm_pre = Linear(self.num_skill*(emb_dim+self.ntotal), emb_dim)

        self.conceptIntergation = conceptIntegration

    def forward(self, concepts, rgaps,  pcounts, acounts):
        concepts = concepts.to(device)
        rgaps,pcounts, acounts = rgaps.to(device), pcounts.to(
            device), acounts.to(device)
        rgap_eye,  pcount_eye, acount_eye = self.rgap_eye.to(device), self.pcount_eye.to(device), self.acount_eye.to(device)
        batch, len, max_concept = concepts.shape
        forget_len = self.forget_len
        # 1.得到 concept_true 每个问题对应的概念embedding
        concept_true = self.conceptIntergation(concepts)
        # 2.得到每个问题对应概念的遗忘embedding
        # 将 rgaps    sgaps   都变成[batch,len,max_concepts,len(forget)
        # print(concepts.shape)
        rgaps = rgaps.unsqueeze(3).repeat(1, 1, 1, forget_len)
        # 进入对应的eye 之后reshape
        rgaps,  pcounts, acounts = rgap_eye[rgaps], pcount_eye[pcounts], acount_eye[acounts]
        emb_rgaps = rgaps.reshape([batch, len, max_concept, -1])
        emb_pcounts = pcounts.reshape([batch, len, max_concept, -1])
        emb_acounts = acounts.reshape([batch, len, max_concept, -1])
        forget_vector1 = torch.cat((emb_rgaps, emb_pcounts, emb_acounts),
                                   -1)  # [batch,len-1,max_concept,ntotal]
        #

        forget_vector2 = torch.zeros((batch, len, self.num_skill, self.ntotal))  # [batch,len,num_skill, ntotal]

        for batch_idx in range(concepts.shape[0]):  # [batch,len,max_concept]
            for len_idx in range(concepts.shape[1]):
                for feature_idx in range(concepts.shape[2]):
                    element_value = concepts[batch_idx, len_idx, feature_idx].item()
                    try:
                        if element_value != -1:
                            forget_vector2[batch_idx, len_idx, element_value] = forget_vector1[
                                batch_idx, len_idx, feature_idx]
                    except:
                        print(f"Batch {batch_idx}, Length {len_idx}, Feature {feature_idx}: {element_value}")

        forget_vector2 = forget_vector2.to(device)
        # 3.integration
        Cct = self.cemb(forget_vector2)  # [batch,len,num_skill,emb_dim]
        theta = torch.mul(concept_true, Cct).to(device)
        theta = torch.cat((theta, forget_vector2), -1)  # [batch,len,num_skill,emb_dim+ntotal]
        theta=theta.reshape(batch,len,-1)
        theta = self.lstm_pre(theta)

        # theta = torch.zeros((batch, len, self.emb_dim)).to(device)
        return theta  # [batch,len,num_skill,emb_dim]
