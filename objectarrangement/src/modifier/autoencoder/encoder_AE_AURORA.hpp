//
// Created by Luca Grillotti
//

#ifndef AE_AURORA_ENCODER_HPP
#define AE_AURORA_ENCODER_HPP

struct EncoderImpl : torch::nn::Module {
    EncoderImpl(int en_hid_dim1, int en_hid_dim2, int en_hid_dim3, int latent_dim) :
        m_conv_1(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, en_hid_dim1, 6))),
        m_conv_s1(torch::nn::Conv2d(torch::nn::Conv2dOptions(en_hid_dim1, en_hid_dim1, 5).stride(2))),
        m_conv_2(torch::nn::Conv2d(torch::nn::Conv2dOptions(en_hid_dim1, en_hid_dim2, 5))),
        m_conv_s2(torch::nn::Conv2d(torch::nn::Conv2dOptions(en_hid_dim2, en_hid_dim2, 4).stride(2))),
        m_conv_3(torch::nn::Conv2d(torch::nn::Conv2dOptions(en_hid_dim2, en_hid_dim3, 3))),
        m_conv_4(torch::nn::Conv2d(torch::nn::Conv2dOptions(en_hid_dim3, latent_dim, 3))),
        m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
        {
            register_module("conv_1", m_conv_1);
            register_module("conv_2", m_conv_2);
            register_module("conv_3", m_conv_3);
            register_module("conv_4", m_conv_4);
            register_module("conv_s1", m_conv_s1);
            register_module("conv_s2", m_conv_s2);
            _initialise_weights();
        }

        torch::Tensor forward(const torch::Tensor &x, torch::Tensor &tmp1, torch::Tensor &tmp2)
        {
            return m_conv_4(torch::relu(m_conv_3(torch::relu(m_conv_s2(
                torch::relu(m_conv_2(torch::relu(m_conv_s1(
                    torch::relu(m_conv_1(x.reshape({-1, 1, static_cast<int>(sqrt(x.size(1))), static_cast<int>(sqrt(x.size(1)))})))))))))))).reshape({x.size(0), -1});
        }

        // https://github.com/pytorch/vision/blob/master/torchvision/csrc/models/googlenet.cpp#L150
        void _initialise_weights()
        {
            for (auto& module : modules(/*include_self=*/false)) 
            {
                if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
                torch::nn::init::kaiming_normal_(M->weight, 0., torch::kFanIn, torch::kReLU);
            }
        }


        torch::nn::Conv2d m_conv_1, m_conv_s1, m_conv_2, m_conv_s2, m_conv_3, m_conv_4;
        torch::Device m_device;
};

TORCH_MODULE(Encoder);

#endif //EXAMPLE_PYTORCH_ENCODER_HPP
