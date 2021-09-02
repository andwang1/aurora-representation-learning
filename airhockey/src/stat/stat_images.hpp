//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_IMG_HPP
#define SFERES2_STAT_IMG_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(Images, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() == Params::pop::nb_gen - 1))
                {
                   std::string prefix = "images_" + boost::lexical_cast<std::string>(ea.gen());
                    _write_images(prefix, ea);
                }
            }

            template<typename EA>
            void _write_images(const std::string &prefix, const EA &ea) const {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                                // retrieve all phenotypes and trajectories                
                matrix_t gen, img;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_geno(ea.pop(), gen);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_image(ea.pop(), img);
                
                matrix_t descriptors, recon_loss, recon_loss_unred, reconstruction, L2_loss, KL_loss, encoder_var, decoder_var;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_stats(gen, img, descriptors, reconstruction, recon_loss, recon_loss_unred, 
                                                                    L2_loss, KL_loss, encoder_var, decoder_var, true);

                // retrieve images where random trajectories are marked differently
                matrix_t contrasted_images(ea.pop().size(), Params::nov::discretisation * Params::nov::discretisation);

                for (int i{0}; i < ea.pop().size(); ++i)
                {
                    auto block = contrasted_images.block<1, Params::nov::discretisation * Params::nov::discretisation>(i, 0);
                    ea.pop()[i]->fit().generate_contrasted_image(block);
                }
                                                        
                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");

                ofs << "FORMAT: INDIV_INDEX, TYPE, DATA\n";
                for (int i{0}; i < reconstruction.rows(); ++i)
                {
                    ofs << i << ", RECON," <<  reconstruction.row(i).format(CommaInitFmt) << "\n";
                    ofs << i << ", ACTUAL," <<  contrasted_images.row(i).format(CommaInitFmt) << "\n";
                    ofs << i << ", L2_LOSS," <<  L2_loss.row(i).format(CommaInitFmt) << "\n";
                    ofs << i << ", RECON_LOSS," <<  recon_loss_unred.row(i).format(CommaInitFmt) << "\n";
                    #ifdef VAE
                    ofs << i << ", KL_LOSS," <<  KL_loss.row(i).format(CommaInitFmt) << "\n";
                    ofs << i << ", DECODER_VAR," <<  decoder_var.row(i).format(CommaInitFmt) << "\n";
                    #endif
                }
            }
        };
    }
}

#endif
