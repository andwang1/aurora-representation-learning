//
// Created by Luca Grillotti
//

#ifndef AE_ENCODER_HPP
#define AE_ENCODER_HPP

struct EncoderImpl : torch::nn::Module {
    EncoderImpl(int input_dim, int en_hid_dim1, int en_hid_dim2, int latent_dim) :
        m_linear_1(torch::nn::Linear(torch::nn::LinearOptions(input_dim, en_hid_dim1))),
        m_linear_2(torch::nn::Linear(torch::nn::LinearOptions(en_hid_dim1, en_hid_dim2))),
        m_linear_3(torch::nn::Linear(torch::nn::LinearOptions(en_hid_dim2, latent_dim))),
        m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
        {
            register_module("linear_1", m_linear_1);
            register_module("linear_2", m_linear_2);
            register_module("linear_3", m_linear_3);
            _initialise_weights();
        }

        torch::Tensor forward(const torch::Tensor &x, torch::Tensor &tmp1, torch::Tensor &tmp2)
        {
            return m_linear_3(torch::relu(m_linear_2(torch::relu(m_linear_1(x)))));
        }

        void _initialise_weights()
        {
            for (auto& module : modules(/*include_self=*/false)) 
            {
                if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get()))
                torch::nn::init::xavier_normal_(M->weight);
            }
        }


        torch::nn::Linear m_linear_1, m_linear_2, m_linear_3;
        torch::Device m_device;
};

TORCH_MODULE(Encoder);

#endif //EXAMPLE_PYTORCH_ENCODER_HPP
