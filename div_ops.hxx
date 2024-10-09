/*
  Finite volume discretisations of divergence operators
 
  ***********

    Copyright B.Dudson, J.Leddy, University of York, September 2016
              email: benjamin.dudson@york.ac.uk

    This file is part of Hermes.

    Hermes is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Hermes is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Hermes.  If not, see <http://www.gnu.org/licenses/>.
  
 */

#ifndef __DIV_OPS_H__
#define __DIV_OPS_H__

#include <field3d.hxx>
#include <vector3d.hxx>

class CustomStencil {
public:
  CustomStencil(Mesh &mesh, const std::string &name,
                const std::string &type = "3x3");
  CustomStencil &operator*=(BoutReal fac);
  CustomStencil &operator/=(BoutReal fac) { return (*this) *= 1 / fac; };
  Field3D apply(const Field3D &a, const std::string &region = "RGN_NOBNDRY");
  BoutReal apply(const Field3D &a, const Ind3D &i);
  BoutReal operator()(const Field3D &a, const Ind3D &i) { return apply(a, i); }

private:
  std::vector<Field3D> coefs;
  std::vector<int> xoffset;
  std::vector<int> zoffset;
};

/*!
 * Diffusion in index space
 * 
 * Similar to using Div_par_diffusion(SQ(mesh->dy)*mesh->g_22, f)
 *
 * @param[in] The field to be differentiated
 * @param[in] bndry_flux  Are fluxes through the boundary calculated?
 */

const Field3D Div_f_v_no_y(const Field3D& n_in, const Field3D& vx,const Field3D& vz, bool bndry_flux);

const Field3D Div_par_diffusion_index(const Field3D &f, bool bndry_flux=true);

const Field3D Div_n_bxGrad_f_B_XPPM(const Field3D &n, const Field3D &f, bool bndry_flux=true, bool poloidal=false, bool positive=false);

const Field3D Div_Perp_Lap_FV_Index(const Field3D &a, const Field3D &f, bool xflux);

// 4th-order flux conserving term, in index space
const Field3D D4DX4_FV_Index(const Field3D &f, bool bndry_flux=false);

// 4th order Z derivative in index space
const Field3D D4DZ4_Index(const Field3D &f);

// Div ( k * Grad(f) )
const Field2D Laplace_FV(const Field2D &k, const Field2D &f);

namespace FCI {
Field3D Div_a_Grad_perp(const Field3D &a, const Field3D &f);

class dagp {
public:
  Field3D operator()(const Field3D &a, const Field3D &f,
                     const std::string &region = "RGN_NOBNDRY");
  dagp(Mesh &mesh);
  dagp &operator*=(BoutReal fac) {
    R /= fac;
    ddR *= fac;
    ddZ *= fac;
    delp2 *= fac * fac;
    return *this;
  }
  dagp &operator/=(BoutReal fac) { return operator*=(1 / fac); }

private:
  Field3D R;
  CustomStencil ddR, ddZ, delp2;
};

class dagp_fv {
public:
  Field3D operator()(const Field3D &a, const Field3D &f);
  dagp_fv(Mesh &mesh);
  dagp_fv &operator*=(BoutReal fac) {
    volume /= fac * fac;
    return *this;
  }
  dagp_fv &operator/=(BoutReal fac) { return operator*=(1 / fac); }

private:
  Field3D fac_XX;
  Field3D fac_XZ;
  Field3D fac_ZX;
  Field3D fac_ZZ;
  Field3D volume;
  BoutReal xflux(const Field3D &a, const Field3D &f, const Ind3D &i);
  BoutReal zflux(const Field3D &a, const Field3D &f, const Ind3D &i);
};
} // namespace FCI
Field3D Div_a_Grad_perp_nonorthog(const Field3D& a, const Field3D& f);

#endif //  __DIV_OPS_H__
