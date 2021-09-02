//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_AE_LOSSES_HPP
#define SFERES2_STAT_AE_LOSSES_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(Losses, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void refresh(EA &ea) 
            {
                std::string prefix = "ae_loss";
                _write_losses(prefix, ea);
            }
            

            template<typename EA>
            void _write_losses(const std::string &prefix, const EA &ea) {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                #ifndef AURORA
                // generate missing observations
                boost::fusion::at_c<0>(ea.fit_modifier()).generate_observations(ea.pop());
                #else // if AURORA then all normal observations aare already generated, need only addditionally the undisturbed ones
                boost::fusion::at_c<0>(ea.fit_modifier()).generate_undisturbed_observations(ea.pop());
                #endif

                matrix_t descriptors, recon_loss, recon_loss_unred, reconstruction, L2_loss, KL_loss, encoder_var, decoder_var;
                {
                    matrix_t gen, img;
                    boost::fusion::at_c<0>(ea.fit_modifier()).get_geno(ea.pop(), gen);
                    boost::fusion::at_c<0>(ea.fit_modifier()).get_image(ea.pop(), img);
                    boost::fusion::at_c<0>(ea.fit_modifier()).get_stats(gen, img, descriptors, reconstruction, recon_loss, recon_loss_unred, 
                                                                        L2_loss, KL_loss, encoder_var, decoder_var, true);
                }

                std::ofstream ofs(fname.c_str(), std::ofstream::app);
                ofs.precision(17);
                double recon = recon_loss.mean();
                double L2 = L2_loss.rowwise().sum().mean();

                // retrieve images without any interference from random observations
                matrix_t undisturbed_images(ea.pop().size(), Params::nov::discretisation * Params::nov::discretisation);
                for (size_t i{0}; i < ea.pop().size(); ++i)
                {undisturbed_images.row(i) = ea.pop()[i]->fit().get_undisturbed_image();}
                
                double L2_undisturbed = (undisturbed_images - reconstruction).array().square().rowwise().sum().mean();

                #ifdef VAE
                float sne_loss{-99};
                if ((boost::fusion::at_c<0>(ea.fit_modifier()).is_train_gen()) && (Params::ae::add_sne_criterion != Params::ae::sne::NoSNE))
                {
                    torch::NoGradGuard no_grad;
                    int num_batches{0};
                    size_t row_index{0};

                    if (reconstruction.rows() > Params::ae::batch_size)
                    {
                        while (row_index < reconstruction.rows() - 1)
                        {
                            int batch_size = (row_index + Params::ae::batch_size < reconstruction.rows()) ? Params::ae::batch_size : (reconstruction.rows() - row_index);
                            torch::Tensor reconstruction_tensor, descriptors_tensor;
                            boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_torch_tensor_from_eigen_matrix(reconstruction.middleRows(row_index, batch_size), reconstruction_tensor);
                            boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_torch_tensor_from_eigen_matrix(descriptors.middleRows(row_index, batch_size), descriptors_tensor);

                            if (torch::cuda::is_available())
                            {
                                reconstruction_tensor = reconstruction_tensor.to(torch::device(torch::kCUDA));
                                descriptors_tensor = descriptors_tensor.to(torch::device(torch::kCUDA));
                            }

                            // get the high dimensional similarities
                            torch::Tensor h_dist_mat, h_variances;
                            boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_sq_dist_matrix(reconstruction_tensor, h_dist_mat);
                            boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_var_from_perplexity(h_dist_mat, h_variances);

                            // similarity matrix, unsqueeze so division is along columns
                            torch::Tensor exp_h_sim_mat = torch::exp(-h_dist_mat / h_variances.unsqueeze(1));

                            // here need to mask out the index i as per TSNE paper
                            torch::Tensor p_j_i = exp_h_sim_mat / (torch::sum(exp_h_sim_mat, {1}) - 1 + 1e-16).unsqueeze(1);

                            // set diagonal to zero as only interested in pairwise similarities, as per TSNE paper
                            p_j_i.fill_diagonal_(0);

                            // get the low dimensional similarities
                            torch::Tensor l_dist_mat;
                            boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_sq_dist_matrix(descriptors_tensor, l_dist_mat);

                            if (Params::ae::add_sne_criterion == Params::ae::sne::TSNE)
                            {
                                torch::Tensor p_ij = (p_j_i + p_j_i.transpose(0, 1)) / (2 * p_j_i.size(0));
                                
                                torch::Tensor l_sim_mat = 1 / (1 + l_dist_mat);

                                // here need to mask out the index i as per TSNE paper, ith term will be = 1 as dist = 0
                                torch::Tensor q_ij = l_sim_mat / (torch::sum(l_sim_mat, {1}) - 1 + 1e-16).unsqueeze(1);
                                
                                // set diagonal to zero as only interested in pairwise similarities, as per TSNE paper
                                q_ij.fill_diagonal_(0);

                                torch::Tensor tsne = p_ij * torch::log((p_ij + 1e-16) / (q_ij  + 1e-16));

                                // set coefficient to dimensionality of data as per VAE-SNE paper
                                sne_loss += (torch::sum(tsne) * reconstruction_tensor.size(1) / batch_size).item<float>();
                            }
                            else if (Params::ae::add_sne_criterion == Params::ae::sne::SNE)
                            {
                                torch::Tensor exp_l_sim_mat = torch::exp(-l_dist_mat);

                                // here need to mask out the index i as per the paper
                                torch::Tensor q_ij = exp_l_sim_mat / (torch::sum(exp_l_sim_mat, {1}) - 1 + 1e-16).unsqueeze(1);
                                // set diagonal to zero as only interested in pairwise similarities, as per TSNE paper
                                q_ij.fill_diagonal_(0);

                                torch::Tensor sne = p_j_i * torch::log((p_j_i + 1e-16) / (q_ij + 1e-16));
                
                                // set coefficient to dimensionality of data as per VAE-SNE paper
                                sne_loss = (torch::sum(sne) * reconstruction_tensor.size(1) / batch_size).item<float>();
                            }
                            row_index += batch_size;
                            ++num_batches;
                        }
                    }
                    sne_loss /= num_batches;
                    // loop end
                }

                // these three are unreduced, need row wise sum and then mean
                double KL = KL_loss.rowwise().sum().mean();
                double en_var = encoder_var.rowwise().sum().mean();
                double de_var = decoder_var.rowwise().sum().mean();

                ofs << ea.gen() << ", " << recon << ", " << KL << ", " << en_var << ", " << de_var << ", " << L2 << ", " << L2_undisturbed;

                #else // AURORA
                ofs << ea.gen() << ", " << recon << ", " << L2 << ", " << L2_undisturbed;
                #endif

                if (boost::fusion::at_c<0>(ea.fit_modifier()).is_train_gen())
                {
                    #ifdef VAE
                    ofs << ", " << sne_loss;
                    #endif
                    ofs << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_random_extension_ratio() << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_epochs_trained() << "/" << Params::ae::nb_epochs << ", IS_TRAIN";
                }
                ofs << "\n";
            }
        };

    }
}


#endif
