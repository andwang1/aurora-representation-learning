//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_NOTMOVEDVAR_HPP
#define SFERES2_STAT_NOTMOVEDVAR_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(NotMovedReconVar, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_diversity == 0)) 
                {
                   std::string prefix = "notmovedvar" + boost::lexical_cast<std::string>(ea.gen());
                    _write_notmovedvar(prefix, ea);
                }
            }

            template<typename EA>
            void _write_notmovedvar(const std::string &prefix, const EA &ea) const {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                // retrieve all phenotypes and trajectories                
                matrix_t gen, img;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_geno(ea.pop(), gen);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_image(ea.pop(), img);
                
                // get all data
                matrix_t descriptors, recon_loss, recon_loss_unred, reconstruction, L2_loss, KL_loss, encoder_var, decoder_var;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_stats(gen, img, descriptors, reconstruction, recon_loss, recon_loss_unred, 
                                                                    L2_loss, KL_loss, encoder_var, decoder_var, true);
                
                matrix_t recon_not_moved(ea.pop().size(), Params::nov::discretisation * Params::nov::discretisation);
                size_t not_moved_counter{0};
                for (int i{0}; i < ea.pop().size(); ++i)
                {
                    if (!(ea.pop()[i]->fit().moved()))
                    {
                        recon_not_moved.row(not_moved_counter) = reconstruction.row(i);
                        ++not_moved_counter;
                    }
                }

                Eigen::VectorXf variance_not_moved = (recon_not_moved.block(0, 0, not_moved_counter, Params::nov::discretisation * Params::nov::discretisation).rowwise() - recon_not_moved.block(0, 0, not_moved_counter, Params::nov::discretisation * Params::nov::discretisation).colwise().mean()).array().square().colwise().sum().array() / (not_moved_counter - 1);
                
                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
                ofs << "Mean Var of Reconstruction of not moved\nVar\n" << variance_not_moved.mean() << "\n" << variance_not_moved.format(CommaInitFmt);
            }
        };
    }
}

#endif
