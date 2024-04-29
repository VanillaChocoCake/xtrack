// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_OCTUPOLE_H
#define XTRACK_OCTUPOLE_H

/*gpufun*/
void Octupole_track_local_particle(
        OctupoleData el,
        LocalParticle* part0
) {
    double length = OctupoleData_get_length(el);
    double backtrack_sign = 1;

    #ifdef XSUITE_BACKTRACK
        length = -length;
        backtrack_sign = -1;
    #endif

    double const k3 = OctupoleData_get_k3(el);
    double const k3s = OctupoleData_get_k3s(el);

    double const knl_oct[4] = {0., 0., 0., backtrack_sign * k3 * length};
    double const ksl_oct[4] = {0., 0., 0., backtrack_sign * k3s * length};

    const int64_t order = OctupoleData_get_order(el);
    const double inv_factorial_order = OctupoleData_get_inv_factorial_order(el);
    /*gpuglmem*/ const double *knl = OctupoleData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = OctupoleData_getp1_ksl(el, 0);

    //start_per_particle_block (part0->part)

        // Drift
        Drift_single_particle(part, length / 2.);

        Multipole_track_single_particle(part,
            0., length, 1, // weight 1
            knl, ksl, order, inv_factorial_order,
            knl_oct, ksl_oct, 2, 0.5,
            backtrack_sign,
            0, 0,
            NULL, NULL, NULL,
            NULL, NULL, NULL,
            NULL, NULL);

        // Drift
        Drift_single_particle(part, length / 2.);

    //end_per_particle_block


}

#endif // XTRACK_OCTUPOLE_H