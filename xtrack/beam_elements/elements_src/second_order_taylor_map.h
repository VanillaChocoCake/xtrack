// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SECONDORDERTAYLORMAP_H
#define XTRACK_SECONDORDERTAYLORMAP_H

/*gpufun*/
void SecondOrderTaylorMap_track_local_particle(SecondOrderTaylorMapData el,
                                               LocalParticle* part0){


    //start_per_particle_block (part0->part)

        double z_in[6];
        double z_out[6];

        z_in[0] = LocalParticle_get_x(part);
        z_in[1] = LocalParticle_get_px(part);
        z_in[2] = LocalParticle_get_y(part);
        z_in[3] = LocalParticle_get_py(part);
        z_in[4] = LocalParticle_get_zeta(part);
        z_in[5] = LocalParticle_get_ptau(part) / LocalParticle_get_beta0(part);

        for (int ii = 0; ii < 6; ii++){
            z_out[ii] = z_in[ii];
        }

        for (int ii = 0; ii < 6; ii++){
            for (int jj = 0; jj < 6; jj++){
                z_out[ii] += SecondOrderTaylorMapData_get_k(el, ii);
            }

        }

        for (int ii = 0; ii < 6; ii++){
            for (int jj = 0; jj < 6; jj++){
                z_out[ii] += SecondOrderTaylorMapData_get_R(el, ii, jj) * z_in[jj];
            }
        }

        for (int ii = 0; ii < 6; ii++){
            for (int jj = 0; jj < 6; jj++){
                for (int kk = 0; kk < 6; kk++){
                    z_out[ii] += SecondOrderTaylorMapData_get_T(el, ii, jj, kk) * z_in[jj] * z_in[kk];
                }
            }
        }

    //end_per_particle_block


    }

#endif