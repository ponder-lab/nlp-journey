# coding=utf-8
# created by msgi on 2020/4/1 7:23 下午
import tensorflow as tf


# simple attention mechanism
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, max_len, hidden_size)
        # hidden_with_time_axis shape: (batch_size，1，hidden_size)
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape: (batch_size，max_len，1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context vector shape: (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)

    def call(self, query, values):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, max_len, hidden_size)
        # score shape: (batch_size, 1, max_len)
        # hidden_with_time_axis shape: (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = tf.matmul(hidden_with_time_axis, self.W1(values), transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=2)
        context_vector = tf.matmul(attention_weights, values)
        return context_vector, attention_weights


class VanillaRNNAttention(tf.keras.layers.Layer):
    def __init__(self, attention_size):
        self.attention_size = attention_size
        self.W = tf.keras.layers.Dense(attention_size, activation='tanh')
        self.U = tf.keras.layers.Dense(1)
        super(VanillaRNNAttention, self).__init__()

    def call(self, x, mask=None):
        # et shape: (batch_size, max_len, attention_size)
        et = self.W(x)
        # at shape: (batch_size, max_len)
        at = tf.nn.softmax(tf.squeeze(self.U(et), axis=-1))
        if mask is not None:
            at *= tf.cast(mask, tf.float32)
        # atx shape: (batch_size, max_len, 1)
        atx = tf.expand_dims(at, -1)

        # sum result shape: (batch_size, attention_size)
        sum_result = tf.reduce_sum(atx * x, axis=1)
        return sum_result

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


# 多头自注意力机制
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        """计算注意力权重。
          q, k, v 必须具有匹配的前置维度。
          k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
          虽然 mask 根据其类型（填充或前瞻）有不同的形状，
          但是 mask 必须能进行广播转换以便求和。
          参数:
            q: 请求的形状 == (..., seq_len_q, depth)
            k: 主键的形状 == (..., seq_len_k, depth)
            v: 数值的形状 == (..., seq_len_v, depth_v)
            mask: Float 张量，其形状能转换成
                  (..., seq_len_q, seq_len_k)。默认为None。
          返回值:
            输出，注意力权重
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # 缩放 matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # 将 mask 加入到缩放的张量上。
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax 在最后一个轴（seq_len_k）上归一化，因此分数相加等于1。
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        """
        分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        """
        :param d_model:
        :param dff:
        :return: A feed forward nn to be stacked after each attention layer.
        """
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


if __name__ == '__main__':
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out.shape, attn.shape)
