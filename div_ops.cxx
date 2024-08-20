/*
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

#include <mpi.h>

#include "div_ops.hxx"

#include <bout/fv_ops.hxx>

#include <bout/assert.hxx>
#include <bout/mesh.hxx>
#include <derivs.hxx>
#include <difops.hxx>
#include <globals.hxx>
#include <output.hxx>
#include <utils.hxx>

#include <bout/index_derivs.hxx>

#include <cmath>

using bout::globals::mesh;

const Field3D Div_par_diffusion_index(const Field3D &f, bool bndry_flux) {
  Field3D result;
  result = 0.0;

  Coordinates *coord = mesh->getCoordinates();
  
  for (int i = mesh->xstart; i <= mesh->xend; i++)
    for (int j = mesh->ystart - 1; j <= mesh->yend; j++)
      for (int k = 0; k < mesh->LocalNz; k++) {
        // Calculate flux at upper surface

        if (!bndry_flux && !mesh->periodicY(i)) {
          if ((j == mesh->yend) && mesh->lastY(i))
            continue;

          if ((j == mesh->ystart - 1) && mesh->firstY(i))
            continue;
        }
        BoutReal J =
	  0.5 * (coord->J(i, j, k) + coord->J(i, j + 1, k)); // Jacobian at boundary

        BoutReal gradient = f(i, j + 1, k) - f(i, j, k);

        BoutReal flux = J * gradient;

        result(i, j, k) += flux / coord->J(i, j, k);
        result(i, j + 1, k) -= flux / coord->J(i, j + 1, k);
      }
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// XPPM methods

BoutReal BOUTMIN(const BoutReal &a, const BoutReal &b, const BoutReal &c,
                 const BoutReal &d) {
  BoutReal r1 = (a < b) ? a : b;
  BoutReal r2 = (c < d) ? c : d;
  return (r1 < r2) ? r1 : r2;
}

struct Stencil1D {
  // Cell centre values
  BoutReal c, m, p, mm, pp;

  // Left and right cell face values
  BoutReal L, R;
};

// First order upwind for testing
void Upwind(Stencil1D &n) { n.L = n.R = n.c; }

// Fromm method
void Fromm(Stencil1D &n) {
  n.L = n.c - 0.25 * (n.p - n.m);
  n.R = n.c + 0.25 * (n.p - n.m);
}

/// The minmod function returns the value with the minimum magnitude
/// If the inputs have different signs then returns zero
BoutReal minmod(BoutReal a, BoutReal b) {
  if (a * b <= 0.0)
    return 0.0;

  if (fabs(a) < fabs(b))
    return a;
  return b;
}

BoutReal minmod(BoutReal a, BoutReal b, BoutReal c) {
  // If any of the signs are different, return zero gradient
  if ((a * b <= 0.0) || (a * c <= 0.0)) {
    return 0.0;
  }

  // Return the minimum absolute value
  return SIGN(a) * BOUTMIN(fabs(a), fabs(b), fabs(c));
}

void MinMod(Stencil1D &n) {
  // Choose the gradient within the cell
  // as the minimum (smoothest) solution
  BoutReal slope = minmod(n.p - n.c, n.c - n.m);
  n.L = n.c - 0.5 * slope; // 0.25*(n.p - n.m);
  n.R = n.c + 0.5 * slope; // 0.25*(n.p - n.m);
}

// Monotonized Central limiter (Van-Leer)
void MC(Stencil1D &n) {
  BoutReal slope =
      minmod(2. * (n.p - n.c), 0.5 * (n.p - n.m), 2. * (n.c - n.m));
  n.L = n.c - 0.5 * slope;
  n.R = n.c + 0.5 * slope;
}

void XPPM(Stencil1D &n, const BoutReal h) {
  // 4th-order PPM interpolation in X

  const BoutReal C = 1.25; // Limiter parameter

  BoutReal h2 = h * h;

  n.R = (7. / 12) * (n.c + n.p) - (1. / 12) * (n.m + n.pp);
  n.L = (7. / 12) * (n.c + n.m) - (1. / 12) * (n.mm + n.p);

  // Apply limiters
  if ((n.c - n.R) * (n.p - n.R) > 0.0) {
    // Calculate approximations to second derivative

    BoutReal D2 = (3. / h2) * (n.c - 2 * n.R + n.p);
    BoutReal D2L = (1. / h2) * (n.m - 2 * n.c + n.p);
    BoutReal D2R = (1. / h2) * (n.c - 2. * n.p + n.pp);

    BoutReal D2lim; // Value to be used in limiter

    // Check if they all have the same sign
    if ((D2 * D2L > 0.0) && (D2 * D2R > 0.0)) {
      // Same sign

      D2lim = SIGN(D2) * BOUTMIN(C * fabs(D2L), C * fabs(D2R), fabs(D2));
    } else {
      // Different sign
      D2lim = 0.0;
    }

    n.R = 0.5 * (n.c + n.p) - (h2 / 6) * D2lim;
  }

  if ((n.m - n.L) * (n.c - n.L) > 0.0) {
    // Calculate approximations to second derivative

    BoutReal D2 = (3. / h2) * (n.m - 2 * n.L + n.c);
    BoutReal D2L = (1. / h2) * (n.mm - 2 * n.m + n.c);
    BoutReal D2R = (1. / h2) * (n.m - 2. * n.c + n.p);

    BoutReal D2lim; // Value to be used in limiter

    // Check if they all have the same sign
    if ((D2 * D2L > 0.0) && (D2 * D2R > 0.0)) {
      // Same sign

      D2lim = SIGN(D2) * BOUTMIN(C * fabs(D2L), C * fabs(D2R), fabs(D2));
    } else {
      // Different sign
      D2lim = 0.0;
    }

    n.L = 0.5 * (n.m + n.c) - (h2 / 6) * D2lim;
  }

  if (((n.R - n.c) * (n.c - n.L) <= 0.0) ||
      ((n.m - n.c) * (n.c - n.p) <= 0.0)) {
    // At a local maximum or minimum

    BoutReal D2 = (6. / h2) * (n.L - 2. * n.c + n.R);

    if (fabs(D2) < 1e-10) {
      n.R = n.L = n.c;
    } else {
      BoutReal D2C = (1. / h2) * (n.m - 2. * n.c + n.p);
      BoutReal D2L = (1. / h2) * (n.mm - 2 * n.m + n.c);
      BoutReal D2R = (1. / h2) * (n.c - 2. * n.p + n.pp);

      BoutReal D2lim;
      // Check if they all have the same sign
      if ((D2 * D2C > 0.0) && (D2 * D2L > 0.0) && (D2 * D2R > 0.0)) {
        // Same sign

        D2lim = SIGN(D2) *
                BOUTMIN(C * fabs(D2L), C * fabs(D2R), C * fabs(D2C), fabs(D2));
        n.R = n.c + (n.R - n.c) * D2lim / D2;
        n.L = n.c + (n.L - n.c) * D2lim / D2;
      } else {
        // Different signs
        n.R = n.L = n.c;
      }
    }
  }
}

/* ***USED***
 *  Div (n * b x Grad(f)/B)
 *
 *
 * poloidal   - If true, includes X-Y flows
 * positive   - If true, limit advected quantity (n_in) to be positive
 */
const Field3D Div_n_bxGrad_f_B_XPPM(const Field3D &n, const Field3D &f,
                                    bool bndry_flux, bool poloidal,
                                    bool positive) {
  Field3D result{0.0};

  Coordinates *coord = mesh->getCoordinates();
  
  //////////////////////////////////////////
  // X-Z advection.
  //
  //             Z
  //             |
  //
  //    fmp --- vU --- fpp
  //     |      nU      |
  //     |               |
  //    vL nL        nR vR    -> X
  //     |               |
  //     |      nD       |
  //    fmm --- vD --- fpm
  //

  int nz = mesh->LocalNz;
  for (const auto& ind : f.getRegion("RGN_NOBNDRY")) {
    kp = ind.zp();
    kpp = ind.zpp();
    km = ind.zm();
    kmm = ind.zmm();
      
    // 1) Interpolate stream function f onto corners fmp, fpp, fpm

    BoutReal fmm = 0.25 * (f[ind] + f[ind.xm()] + f[km] +
			   f[km.xm()]);
    BoutReal fmp = 0.25 * (f[ind] + f[kp] + f[ind.xm()] +
			   f[kp.xm()]); // 2nd order accurate
    BoutReal fpp = 0.25 * (f[ind] + f[kp] + f[ind.xp()] +
			   f[kp.xp()]);
    BoutReal fpm = 0.25 * (f[ind] + f[ind.xp()] + f[km] +
			   f[km.xp()]);

    // 2) Calculate velocities on cell faces

    BoutReal vU = 0.5 * (coord->J[ind] + coord->J[kp]) * (fmp - fpp) /
      coord->dx[ind]; // -J*df/dx
    BoutReal vD = 0.5 * (coord->J[ind] + coord->J[km]) * (fmm - fpm) /
      coord->dx[ind]; // -J*df/dx

    BoutReal vR = 0.5 * (coord->J[ind] + coord->J[ind.xp()]) * (fpp - fpm) /
      coord->dz[ind]; // J*df/dz
    BoutReal vL = 0.5 * (coord->J[ind] + coord->J[ind.xm()]) * (fmp - fmm) /
      coord->dz[ind]; // J*df/dz

    // output.write("NEW: (%d,%d,%d) : (%e/%e, %e/%e)\n", i,j,k,vL,vR,
    // vU,vD);

    // 3) Calculate n on the cell faces. The sign of the
    //    velocity determines which side is used.

    // X direction
    Stencil1D s;
    s.c = n[ind];
    s.m = n[ind.xm()];
    s.mm = n[ind.xmm()];
    s.p = n[ind.xp()];
    s.pp = n[ind.xpp()];

    MC(s);

    // Right side
    if ((mesh->lastX()) && (i.x() == mesh->xend)) {
      // At right boundary in X

      if (bndry_flux) {
	BoutReal flux;
	if (vR > 0.0) {
	  // Flux to boundary
	  flux = vR * s.R;
	} else {
	  // Flux in from boundary
	  flux = vR * 0.5 * (n[ind.xp()] + n[ind]);
	}
	result[ind] += flux / (coord->dx[ind] * coord->J[ind]);
	result[ind.xp()] -=
	  flux / (coord->dx[ind.xp()] * coord->J[ind.xp()]);
      }
    } else {
      // Not at a boundary
      if (vR > 0.0) {
	// Flux out into next cell
	BoutReal flux = vR * s.R;
	result[ind] += flux / (coord->dx[ind] * coord->J[ind]);
	result[ind.xp()] -=
	  flux / (coord->dx[ind.xp()] * coord->J[ind.xp()]);
      }
    }

    // Left side

    if ((mesh->firstX()) && (ind.x() == mesh->xstart)) {
      // At left boundary in X

      if (bndry_flux) {
	BoutReal flux;

	if (vL < 0.0) {
	  // Flux to boundary
	  flux = vL * s.L;
	} else {
	  // Flux in from boundary
	  flux = vL * 0.5 * (n[ind.xm()] + n[ind]);
	}
	result[ind] -= flux / (coord->dx[ind] * coord->J[ind]);
	result[ind.xm()] +=
	  flux / (coord->dx[ind.xm()] * coord->J[ind.xm()]);
      }
    } else {
      // Not at a boundary

      if (vL < 0.0) {
	BoutReal flux = vL * s.L;
	result[ind] -= flux / (coord->dx[ind] * coord->J[ind]);
	result[ind.xm()] +=
	  flux / (coord->dx[ind.xm()] * coord->J[ind.xm()]);
      }
    }

    /// NOTE: Need to communicate fluxes

    // Z direction
    s.m = n[km];
    s.mm = n[kmm];
    s.p = n[kp];
    s.pp = n[kpp];

    // Upwind(s, coord->dz);
    // XPPM(s, coord->dz);
    // Fromm(s, coord->dz);
    MC(s);

    if (vU > 0.0) {
      BoutReal flux = vU * s.R; 
      result[ind] += flux / (coord->J[ind] * coord->dz[ind]);
      result[kp] -= flux / (coord->J[kp] * coord->dz[kp]);
    }
    if (vD < 0.0) {
      BoutReal flux = vD * s.L; 
      result[ind] -= flux / (coord->J[ind] * coord->dz[ind]);
      result[km] += flux  / (coord->J[km] * coord->dz[km]);
    }
  }
  FV::communicateFluxes(result);

  //////////////////////////////////////////
  // X-Y advection.
  //
  //
  //  This code does not deal with corners correctly. This may or may not be
  //  important.
  //
  // 1/J d/dx ( J n (g^xx g^yz / B^2) df/dy) - 1/J d/dy( J n (g^xx g^yz / B^2)
  // df/dx )
  //
  // Interpolating stream function f_in onto corners fmm, fmp, fpp, fpm
  // is complicated because the corner point in X-Y is not communicated
  // and at an X-point it is shared with 8 cells, rather than 4
  // (being at the X-point itself)
  // Corners also need to be shifted to the correct toroidal angle

  if (poloidal) {
    // X flux

    Field3D dfdy = DDY(f);
    mesh->communicate(dfdy);
    dfdy.applyBoundary("neumann");

    int xs = mesh->xstart - 1;
    int xe = mesh->xend;
    if (!bndry_flux) {
      // No boundary fluxes
      if (mesh->firstX()) {
        // At an inner boundary
        xs = mesh->xstart;
      }
      if (mesh->lastX()) {
        // At outer boundary
        xe = mesh->xend - 1;
      }
    }

    for (int i = xs; i <= xe; i++)
      for (int j = mesh->ystart - 1; j <= mesh->yend; j++)
        for (int k = 0; k < mesh->LocalNz; k++) {

          // Average dfdy to right X boundary
          BoutReal f_R =
	    0.5 * ((coord->g11(i + 1, j, k) * coord->g23(i + 1, j, k) /
		    SQ(coord->Bxy(i + 1, j, k))) *
		   dfdy(i + 1, j, k) +
		   (coord->g11(i, j, k) * coord->g23(i, j, k) / SQ(coord->Bxy(i, j, k))) *
		   dfdy(i, j, k));

          // Advection velocity across cell face
          BoutReal Vx = 0.5 * (coord->J(i + 1, j, k) + coord->J(i, j, k)) * f_R;

          // Fromm method
          BoutReal flux = Vx;
          if (Vx > 0) {
            // Right boundary of cell (i,j,k)
            BoutReal nval =
                n(i, j, k) + 0.25 * (n(i + 1, j, k) - n(i - 1, j, k));
            if (positive && (nval < 0.0)) {
              // Limit value to positive
              nval = 0.0;
            }
            flux *= nval;
          } else {
            // Left boundary of cell (i+1,j,k)
            BoutReal nval =
                n(i + 1, j, k) - 0.25 * (n(i + 2, j, k) - n(i, j, k));
            if (positive && (nval < 0.0)) {
              nval = 0.0;
            }
            flux *= nval;
          }

          result(i, j, k) += flux / (coord->dx(i, j, k) * coord->J(i, j, k));
          result(i + 1, j, k) -=
	    flux / (coord->dx(i + 1, j, k) * coord->J(i + 1, j, k));
        }
  }

  if (poloidal) {
    // Y flux
    
    Field3D dfdx = DDX(f);
    mesh->communicate(dfdx);
    dfdx.applyBoundary("neumann");

    // This calculation is in field aligned coordinates
    dfdx = toFieldAligned(dfdx);
    Field3D n_fa = toFieldAligned(n);
    
    Field3D yresult{zeroFrom(n_fa)};
    
    for (int i = mesh->xstart; i <= mesh->xend; i++) {
      int ys = mesh->ystart - 1;
      int ye = mesh->yend;

      if (!bndry_flux && !mesh->periodicY(i)) {
        // No boundary fluxes
        if (mesh->firstY(i)) {
          // At an inner boundary
          ys = mesh->ystart;
        }
        if (mesh->lastY(i)) {
          // At outer boundary
          ye = mesh->yend - 1;
        }
      }

      for (int j = ys; j <= ye; j++) {
        for (int k = 0; k < mesh->LocalNz; k++) {
          // Y flow

          // Average dfdx to upper Y boundary
          BoutReal f_U =
	    0.5 * ((coord->g11(i, j + 1, k) * coord->g23(i, j + 1, k) /
		    SQ(coord->Bxy(i, j + 1, k))) *
		   dfdx(i, j + 1, k) +
		   (coord->g11(i, j, k) * coord->g23(i, j, k) / SQ(coord->Bxy(i, j, k))) *
		   dfdx(i, j, k));

          BoutReal Vy = -0.5 * (coord->J(i, j + 1, k) + coord->J(i, j, k)) * f_U;

          if (mesh->firstY(i) && !mesh->periodicY(i) &&
              (j == mesh->ystart - 1)) {
            // Lower y boundary. Allow flows out of the domain only
            if (Vy > 0.0)
              Vy = 0.0;
          }
          if (mesh->lastY(i) && !mesh->periodicY(i) && (j == mesh->yend)) {
            // Upper y boundary
            if (Vy < 0.0)
              Vy = 0.0;
          }

          // Fromm method
          BoutReal flux = Vy;
          if (Vy > 0) {
            // Right boundary of cell (i,j,k)
            BoutReal nval =
                n_fa(i, j, k) + 0.25 * (n_fa(i, j + 1, k) - n_fa(i, j - 1, k));
            if (positive && (nval < 0.0)) {
              nval = 0.0;
            }
            flux *= nval;
          } else {
            // Left boundary of cell (i,j+1,k)
            BoutReal nval =
                n_fa(i, j + 1, k) - 0.25 * (n_fa(i, j + 2, k) - n_fa(i, j, k));
            if (positive && (nval < 0.0)) {
              nval = 0.0;
            }
            flux *= nval;
          }

          yresult(i, j, k) += flux / (coord->dy(i, j, k) * coord->J(i, j, k));
          yresult(i, j + 1, k) -= flux / (coord->dy(i, j + 1, k) * coord->J(i, j + 1, k));
        }
      }
    }
    result += fromFieldAligned(yresult);
  }
  
  return result;
}

/// *** USED ***
const Field3D Div_Perp_Lap_FV_Index(const Field3D &as, const Field3D &fs,
                                    bool xflux) {

  Field3D result = 0.0;

  //////////////////////////////////////////
  // X-Z diffusion in index space
  //
  //            Z
  //            |
  //
  //     o --- gU --- o
  //     |     nU     |
  //     |            |
  //    gL nL      nR gR    -> X
  //     |            |
  //     |     nD     |
  //     o --- gD --- o
  //

  Coordinates *coord = mesh->getCoordinates();
  
  for (int i = mesh->xstart; i <= mesh->xend; i++)
    for (int j = mesh->ystart; j <= mesh->yend; j++)
      for (int k = 0; k < mesh->LocalNz; k++) {
        int kp = (k + 1) % mesh->LocalNz;
        int km = (k - 1 + mesh->LocalNz) % mesh->LocalNz;

        // Calculate gradients on cell faces

        BoutReal gR = fs(i + 1, j, k) - fs(i, j, k);

        BoutReal gL = fs(i, j, k) - fs(i - 1, j, k);

        BoutReal gD = fs(i, j, k) - fs(i, j, km);

        BoutReal gU = fs(i, j, kp) - fs(i, j, k);

        // Flow right
        BoutReal flux = gR * 0.25 * (coord->J(i + 1, j, k) + coord->J(i, j, k)) *
	  (coord->dx(i + 1, j, k) + coord->dx(i, j, k)) *
	  (as(i + 1, j, k) + as(i, j, k));

        result(i, j, k) += flux / (coord->dx(i, j, k) * coord->J(i, j, k));

        // Flow left
        flux = gL * 0.25 * (coord->J(i - 1, j, k) + coord->J(i, j, k)) *
	  (coord->dx(i - 1, j, k) + coord->dx(i, j, k)) *
	  (as(i - 1, j, k) + as(i, j, k));

        result(i, j, k) -= flux / (coord->dx(i, j, k) * coord->J(i, j, k));

        // Flow up

        flux = gU * 0.25 *  (coord->J(i, j, k) + coord->J(i, j, kp)) *
	  (coord->dz(i, j, k) + coord->dz(i, j, kp)) *
	  (as(i, j, k) + as(i, j, kp));
	
        result(i, j, k) += flux / (coord->dz(i, j, k) * coord->J(i, j, k));

	// Flow down
        flux = gD * 0.25 *  (coord->J(i, j, km) + coord->J(i, j, k)) *
	  (coord->dz(i, j, km) + coord->dz(i, j, k)) *
	  (as(i, j, km) + as(i, j, k));

        result(i, j, k) -= flux / (coord->dz(i, j, k) * coord->J(i, j, k));
      }
  
  return result;
}

// *** USED ***
const Field3D D4DX4_FV_Index(const Field3D &f, bool bndry_flux) {
  Field3D result = 0.0;

  Coordinates *coord = mesh->getCoordinates();
  
  for (int i = mesh->xstart; i <= mesh->xend; i++)
    for (int j = mesh->ystart; j <= mesh->yend; j++) {
      for (int k = 0; k < mesh->LocalNz; k++) {

        // 3rd derivative at right boundary

        BoutReal d3fdx3 = (f(i + 2, j, k) - 3. * f(i + 1, j, k) +
                           3. * f(i, j, k) - f(i - 1, j, k));

        BoutReal flux = 0.25 * (coord->dx(i, j, k) + coord->dx(i + 1, j, k)) *
	  (coord->J(i, j, k) + coord->J(i + 1, j, k)) * d3fdx3;

        if (mesh->lastX() && (i == mesh->xend)) {
          // Boundary

          if (bndry_flux) {
            // Use a one-sided difference formula

            d3fdx3 = -((16. / 5) * 0.5 *
                           (f(i + 1, j, k) + f(i, j, k)) // Boundary value f_b
                       - 6. * f(i, j, k)                 // f_0
                       + 4. * f(i - 1, j, k)             // f_1
                       - (6. / 5) * f(i - 2, j, k)       // f_2
                       );

            flux = 0.25 * (coord->dx(i, j, k) + coord->dx(i + 1, j, k)) *
	      (coord->J(i, j, k) + coord->J(i + 1, j, k)) * d3fdx3;

          } else {
            // No fluxes through boundary
            flux = 0.0;
          }
        }

        result(i, j, k) += flux / (coord->J(i, j, k) * coord->dx(i, j, k));
        result(i + 1, j, k) -= flux / (coord->J(i + 1, j, k) * coord->dx(i + 1, j, k));

        if (j == mesh->xstart) {
          // Left cell boundary, no flux through boundaries

          if (mesh->firstX()) {
            // On an X boundary

            if (bndry_flux) {
              d3fdx3 = -(-(16. / 5) * 0.5 *
                             (f(i - 1, j, k) + f(i, j, k)) // Boundary value f_b
                         + 6. * f(i, j, k)                 // f_0
                         - 4. * f(i + 1, j, k)             // f_1
                         + (6. / 5) * f(i + 2, j, k)       // f_2
                         );

              flux = 0.25 * (coord->dx(i, j, k) + coord->dx(i + 1, j, k)) *
		(coord->J(i, j, k) + coord->J(i + 1, j, k)) * d3fdx3;

              result(i, j, k) -= flux / (coord->J(i, j, k) * coord->dx(i, j, k));
              result(i - 1, j, k) +=
		flux / (coord->J(i - 1, j, k) * coord->dx(i - 1, j, k));
            }

          } else {
            // Not on a boundary
            d3fdx3 = (f(i + 1, j, k) - 3. * f(i, j, k) + 3. * f(i - 1, j, k) -
                      f(i - 2, j, k));

            flux = 0.25 * (coord->dx(i, j, k) + coord->dx(i + 1, j, k)) *
	      (coord->J(i, j, k) + coord->J(i + 1, j, k)) * d3fdx3;

            result(i, j, k) -= flux / (coord->J(i, j, k) * coord->dx(i, j, k));
            result(i - 1, j, k) +=
	      flux / (coord->J(i - 1, j, k) * coord->dx(i - 1, j, k));
          }
        }
      }
    }

  return result;
}

const Field3D D4DZ4_Index(const Field3D &f) {
  Field3D result{emptyFrom(f)};

  BOUT_FOR(i, f.getRegion("RGN_NOBNDRY")) {
    result[i] = f[i.zpp()] - 4. * f[i.zp()] + 6. * f[i] - 4. * f[i.zm()] + f[i.zmm()];
  }
  return result;
}

/*! *** USED ***
 * X-Y diffusion
 *
 * NOTE: Assumes g^12 = 0, so X and Y are orthogonal. Otherwise
 * we would need the corner cell values to take Y derivatives along X edges
 *
 */
const Field2D Laplace_FV(const Field2D &k, const Field2D &f) {
  Field2D result;
  result.allocate();

  Coordinates *coord = mesh->getCoordinates();
  Field2D g11_2D = DC(coord->g11);
  Field2D g22_2D = DC(coord->g22);
  Field2D dx_2D = DC(coord->dx);
  Field2D J_2D = DC(coord->J);

  for (int i = mesh->xstart; i <= mesh->xend; i++)
    for (int j = mesh->ystart; j <= mesh->yend; j++) {

      // Calculate gradients on cell faces

      BoutReal gR = (g11_2D(i, j) + g11_2D(i + 1, j)) *
                    (f(i + 1, j) - f(i, j)) /
                    (dx_2D(i + 1, j) + dx_2D(i, j));

      BoutReal gL = (g11_2D(i - 1, j) + g11_2D(i, j)) *
                    (f(i, j) - f(i - 1, j)) /
                    (dx_2D(i - 1, j) + dx_2D(i, j));

      BoutReal gU = (g22_2D(i, j) + g22_2D(i, j + 1)) *
                    (f(i, j + 1) - f(i, j)) /
                    (dx_2D(i, j + 1) + dx_2D(i, j));

      BoutReal gD = (g22_2D(i, j - 1) + g22_2D(i, j)) *
                    (f(i, j) - f(i, j - 1)) /
                    (dx_2D(i, j) + dx_2D(i, j - 1));

      // Flow right

      BoutReal flux = gR * 0.25 * (J_2D(i + 1, j) + J_2D(i, j)) *
                      (k(i + 1, j) + k(i, j));

      result(i, j) = flux / (dx_2D(i, j) * J_2D(i, j));

      // Flow left

      flux = gL * 0.25 * (J_2D(i - 1, j) + J_2D(i, j)) *
             (k(i - 1, j) + k(i, j));
      result(i, j) -= flux / (dx_2D(i, j) * J_2D(i, j));

      // Flow up

      flux = gU * 0.25 * (J_2D(i, j + 1) + J_2D(i, j)) *
             (k(i, j + 1) + k(i, j));
      result(i, j) += flux / (dx_2D(i, j) * J_2D(i, j));

      // Flow down

      flux = gD * 0.25 * (J_2D(i, j - 1) + J_2D(i, j)) *
             (k(i, j - 1) + k(i, j));
      result(i, j) -= flux / (dx_2D(i, j) * J_2D(i, j));
    }
  return result;
}

namespace FCI {
namespace {
template <DIRECTION dir> BoutReal getderiv(Ind3D i, const Field3D &f) {
  static_assert(dir == DIRECTION::X || dir == DIRECTION::Z);
  if (dir == DIRECTION::X) {
    return (f[i.zp()] - f[i.zm()]) * 1;
  }
  return (f[i.xp()] - f[i.xm()]) * 1;
}
template <DIRECTION dir, int sign>
BoutReal getflux(Ind3D i, const Field3D &a, const Field3D &f, const Field3D &di,
                 const Field3D &dj, const Field3D &gii, const Field3D &J,
                 const Field3D &gij, const Field3D &g_jj) {
  Ind3D is;
  if (sign > 0) {
    is = i.plus<sign, dir>();
  } else {
    is = i.minus<-sign, dir>();
  }
  BoutReal ai = a[i] + a[is];
  BoutReal Ji = J[i] + J[is];
  BoutReal dji = dj[i] + dj[is];
  BoutReal df = (f[is] - f[i]) * sign;
  BoutReal dii = di[i] + di[is];
  BoutReal giii = gii[i] + gii[is];
  BoutReal dfj = (getderiv<dir>(i, f) + getderiv<dir>(is, f)) * 0.5;
  BoutReal giji = gij[i] + gij[is];
  // BoutReal g_jji = 0.5 * (g_jj[i] + g_jj[is]);
  return ai * (giii * df / dii + giji * dfj / dji) * Ji * dji * 0.125 * sign;
  // return Ji * ai * dji * (df * giii / dii + dfj * giji / dji) * 0.125 * sign;

  // BoutReal dfi = (gii[i] + gii[is]) * (f[is] - f[i]) * sign / (di[i] +
  // di[is]); BoutReal dfj = (gij[i] + gij[is] ) * (getderiv<dir>(i, f) +
  // getderiv<dir>(is, f)) / (dj[i] + dj[is]) * 0.5; BoutReal Ai = J[i] * dj[i]
  // * gii[i] + J[is] * dj[is] * gii[is]; //sqrt(gjj[i]) * dj[i] + sqrt(gjj[is])
  // * dj[is]; return ai * (dfi + dfj)  * Ai / (8 * -sign);
}
} // namespace
Field3D Div_a_Grad_perp(const Field3D &a, const Field3D &f) {
  ASSERT1_FIELDS_COMPATIBLE(a, f);
#if 0
  auto coord = a.getCoordinates();
  const auto &g11 = coord->g11;
  const auto &g33 = coord->g33;
  const auto &g13 = coord->g13;
  const auto &g_11 = coord->g_11;
  const auto &g_33 = coord->g_33;
  const auto &J = coord->J;
  // Assumes g23 and g12 are zero
  const auto aJ = a * J;
  const auto ddxf = DDX(f);
  const auto ddzf = DDZ(f, CELL_DEFAULT, "DEFAULT", "RGN_NOY");
  auto result = DDX(aJ * g11) * DDX(f) + DDX(aJ * g13 * ddzf) +
                DDZ(aJ * g13 * ddxf) + DDZ(aJ * g33) * DDZ(f);
  result /= J;
  result += a * g11 * D2DX2(f);
  result += a * g33 * D2DZ2(f);
#elif 0
  BoutReal ar = getUniform(a);
  auto result = ar * Delp2(f);
#else
  auto coord = a.getCoordinates();
  auto g11 = coord->g11;
  auto g33 = coord->g33;
  auto g13 = coord->g13;
  auto g_11 = coord->g_11;
  auto g_33 = coord->g_33;
  const auto &J = coord->J;
  // auto g_13 = coord->g_13;
  auto dx = coord->dx;
  auto dz = coord->dz;
  Field3D result = emptyFrom(f);
  BOUT_FOR(i, result.getRegion("RGN_NOBNDRY")) {
    result[i] = getflux<DIRECTION::Z, +1>(i, a, f, dz, dx, g33, J, g13,
                                          g_11); //, g_33);
    result[i] += getflux<DIRECTION::Z, -1>(i, a, f, dz, dx, g33, J, g13,
                                           g_11); //, g_33);
    result[i] += getflux<DIRECTION::X, +1>(i, a, f, dx, dz, g11, J, g13,
                                           g_33); //, g_33);
    result[i] += getflux<DIRECTION::X, -1>(i, a, f, dx, dz, g11, J, g13,
                                           g_33); //, g_33);
    // result[i] += getflux<DIRECTION::Z, -1>(i, a, f, dz, dx, g_11, g_13,
    // g_33); result[i] += getflux<DIRECTION::Z, +1>(i, a, f, dz, dx, g_33,
    // g_13, g_11); result[i] += getflux<DIRECTION::Z, -1>(i, a, f, dz, dx,
    // g_33, g_13, g_11);
    //  auto im = i.xm();
    //  auto ip = i.xp();
    //  result[i] = (a[i] + a[is]) * (f[i] - f[is]) * (sqrt(g_33[i]) * dz[i] +
    //  sqrt(g_33[is]) * dz[is]) / (dx[i] + dx[ip]); result[i] -= (a[ip] + a[i])
    //  * (f[ip] - f[i]) * (sqrt(g_33[ip]) * dz[ip] + sqrt(g_33[i]) * dz[i]) /
    //  (dx[i] + dx[im]); im = i.zm(); ip = i.zp(); result[i] += (a[i] + a[im])
    //  * (f[i] - f[im]) * (sqrt(g_11[i]) * dx[i] + sqrt(g_11[im]) * dx[im]) /
    //  (dz[i] + dz[ip]); result[i] -= (a[ip] + a[i]) * (f[ip] - f[i]) *
    //  (sqrt(g_11[ip]) * dx[ip] + sqrt(g_11[i]) * dx[i]) / (dz[i] + dz[im]);
    result[i] /= J[i] * dx[i] * dz[i]; // / sqrt(coord->g_22[i]);
  }
  // auto result = DDX(a * Jgxx) * DDX(f) + a * Jgxx * D2DX2(f);
  // result += DDX(a * Jgxz * DDZ(f, CELL_DEFAULT, "DEFAULT", "RGN_NOY"));
  // result += DDZ(a * Jgxz * DDX(f));
  // result += DDZ(a * Jgzz) * DDZ(f) + a * Jgzz * D2DZ2(f);
  // result /= J;
#endif
  result.name = "FCI::Div_a_Grad_perp()";
  return result;
}
} // namespace FCI


// Div ( a Grad_perp(f) ) -- ∇⊥ ( a ⋅ ∇⊥ f) --  Vorticity
// Includes nonorthogonal X-Y and X-Z metric corrections
//
Field3D Div_a_Grad_perp_nonorthog(const Field3D& a, const Field3D& f) {
  ASSERT2(a.getLocation() == f.getLocation());

  Mesh* mesh = a.getMesh();

  Coordinates* coord = f.getCoordinates();

  // Y and Z fluxes require Y derivatives

  // Fields containing values along the magnetic field
  Field3D fup(mesh), fdown(mesh);
  Field3D aup(mesh), adown(mesh);

  Field3D g23up(mesh), g23down(mesh);
  Field3D g_23up(mesh), g_23down(mesh);
  Field3D g12up(mesh), g12down(mesh);
  Field3D g_12up(mesh), g_12down(mesh);
  Field3D Jup(mesh), Jdown(mesh);
  Field3D dyup(mesh), dydown(mesh);
  Field3D dzup(mesh), dzdown(mesh);
  Field3D Bxyup(mesh), Bxydown(mesh);

  // Values on this y slice (centre).
  // This is needed because toFieldAligned may modify the field
  Field3D fc = f;
  Field3D ac = a;

  Field3D g23c = coord->g23;
  Field3D g_23c = coord->g_23;
  Field3D g12c = coord->g12;
  Field3D g_12c = coord->g_12;
  Field3D Jc = coord->J;
  Field3D dxc = coord->dx;
  Field3D dyc = coord->dy;
  Field3D dzc = coord->dz;
  Field3D Bxyc = coord->Bxy;

  // Calculate the X derivative at cell edge (X + 1/2), including in Y guard cells
  // This is used to calculate Y flux contribution from g21 * d/dx
  Field3D fddx_xhigh(mesh);
  fddx_xhigh.allocate();
  for (int i = mesh->xstart - 1; i <= mesh->xend; i++) {
    for (int j = mesh->ystart - 1; j <= mesh->yend + 1;
         j++) { // Note: Including one guard cell
      for (int k = 0; k < mesh->LocalNz; k++) {
        fddx_xhigh(i, j, k) = 2. * (f(i + 1, j, k) - f(i, j, k))
                              / (coord->dx(i, j, k) + coord->dx(i + 1, j, k));
      }
    }
  }
  Field3D fddx_xhigh_up(mesh), fddx_xhigh_down(mesh);

  // Result of the Y and Z fluxes
  Field3D yzresult(mesh);
  yzresult.allocate();

  if (f.hasParallelSlices() && a.hasParallelSlices()) {
    // Both inputs have yup and ydown

    fup = f.yup();
    fdown = f.ydown();

    aup = a.yup();
    adown = a.ydown();

    fddx_xhigh.applyBoundary("Neumann");
    mesh->communicate(fddx_xhigh);
    fddx_xhigh.applyParallelBoundary("parallel_neumann_o2");
    fddx_xhigh_up = fddx_xhigh.yup();
    fddx_xhigh_down = fddx_xhigh.ydown();
  } else {
    // At least one input doesn't have yup/ydown fields.
    // Need to shift to/from field aligned coordinates

    fup = fdown = fc = toFieldAligned(f);
    aup = adown = ac = toFieldAligned(a);

    fddx_xhigh_up = fddx_xhigh_down = toFieldAligned(fddx_xhigh);

    yzresult.setDirectionY(YDirectionType::Aligned);
  }

  if (bout::build::use_metric_3d) {
    // 3D Metric, need yup/ydown fields.
    // Requires previous communication of metrics
    // -- should insert communication here?
    if (!coord->g12.hasParallelSlices() || !coord->g_12.hasParallelSlices()
        || !coord->g23.hasParallelSlices() || !coord->g_23.hasParallelSlices()
        || !coord->dy.hasParallelSlices() || !coord->dz.hasParallelSlices()
        || !coord->Bxy.hasParallelSlices() || !coord->J.hasParallelSlices()) {
      throw BoutException("metrics have no yup/down: Maybe communicate in init?");
    }

    g23up = coord->g23.yup();
    g23down = coord->g23.ydown();

    g_23up = coord->g_23.yup();
    g_23down = coord->g_23.ydown();

    g12up = coord->g12.yup();
    g12down = coord->g12.ydown();

    g_12up = coord->g_12.yup();
    g_12down = coord->g_12.ydown();

    Jup = coord->J.yup();
    Jdown = coord->J.ydown();

    dyup = coord->dy.yup();
    dydown = coord->dy.ydown();

    dzup = coord->dz.yup();
    dzdown = coord->dz.ydown();

    Bxyup = coord->Bxy.yup();
    Bxydown = coord->Bxy.ydown();

  } else {
    // No 3D metrics
    // Need to shift to/from field aligned coordinates
    g23up = g23down = g23c = toFieldAligned(coord->g23);
    g_23up = g_23down = g_23c = toFieldAligned(coord->g_23);
    g12up = g12down = g12c = toFieldAligned(coord->g12);
    g_12up = g_12down = g_12c = toFieldAligned(coord->g_12);
    Jup = Jdown = Jc = toFieldAligned(coord->J);
    dxc = toFieldAligned(coord->dx);
    dyup = dydown = dyc = toFieldAligned(coord->dy);
    dzup = dzdown = dzc = toFieldAligned(coord->dz);
    Bxyup = Bxydown = Bxyc = toFieldAligned(coord->Bxy);
  }

  // Y flux
  // Includes fluxes due to Z derivatives (non-zero g23 metric)
  // and due to X derivatives if grid is nonorthogonal (non-zero g12 metric)

  for (int i = mesh->xstart; i <= mesh->xend; i++) {
    for (int j = mesh->ystart; j <= mesh->yend; j++) {
      for (int k = 0; k < mesh->LocalNz; k++) {
        // Calculate flux between j and j+1
        int kp = (k + 1) % mesh->LocalNz;
        int km = (k - 1 + mesh->LocalNz) % mesh->LocalNz;

        BoutReal coef_yz =
            0.5
            * (g_23c(i, j, k) / SQ(Jc(i, j, k) * Bxyc(i, j, k))
               + g_23up(i, j + 1, k) / SQ(Jup(i, j + 1, k) * Bxyup(i, j + 1, k)));

        BoutReal coef_xy =
            0.5
            * (g_12c(i, j, k) / SQ(Jc(i, j, k) * Bxyc(i, j, k))
               + g_12up(i, j + 1, k) / SQ(Jup(i, j + 1, k) * Bxyup(i, j + 1, k)));

        // Calculate Z derivative at y boundary
        BoutReal dfdz =
            0.5 * (fc(i, j, kp) - fc(i, j, km) + fup(i, j + 1, kp) - fup(i, j + 1, km))
            / (dzc(i, j, k) + dzup(i, j + 1, k));

        // Y derivative
        BoutReal dfdy =
            2. * (fup(i, j + 1, k) - fc(i, j, k)) / (dyup(i, j + 1, k) + dyc(i, j, k));

        // X derivative at Y boundary
        BoutReal dfdx = 0.25
                        * (fddx_xhigh(i - 1, j, k) + fddx_xhigh(i, j, k)
                           + fddx_xhigh_up(i - 1, j + 1, k) + fddx_xhigh_up(i, j + 1, k));

        BoutReal fout =
            0.25 * (ac(i, j, k) + aup(i, j + 1, k))
            * (
                // Component of flux from d/dz and d/dy
                (Jc(i, j, k) * g23c(i, j, k) + Jup(i, j + 1, k) * g23up(i, j + 1, k))
                    * (dfdz - coef_yz * dfdy)
                // Non-orthogonal mesh correction with g12 metric
                + (Jc(i, j, k) * g12c(i, j, k) + Jup(i, j + 1, k) * g12up(i, j + 1, k))
                      * (dfdx - coef_xy * dfdy));

        yzresult(i, j, k) = fout / (dyc(i, j, k) * Jc(i, j, k));

        // Calculate flux between j and j-1
        coef_yz =
            0.5
            * (g_23c(i, j, k) / SQ(Jc(i, j, k) * Bxyc(i, j, k))
               + g_23down(i, j - 1, k) / SQ(Jdown(i, j - 1, k) * Bxydown(i, j - 1, k)));

        coef_xy =
            0.5
            * (g_12c(i, j, k) / SQ(Jc(i, j, k) * Bxyc(i, j, k))
               + g_12down(i, j - 1, k) / SQ(Jdown(i, j - 1, k) * Bxydown(i, j - 1, k)));

        dfdz = 0.5
               * (fc(i, j, kp) - fc(i, j, km) + fdown(i, j - 1, kp) - fdown(i, j - 1, km))
               / (dzc(i, j, k) + dzdown(i, j - 1, k));

        dfdy = 2. * (fc(i, j, k) - fdown(i, j - 1, k))
               / (dyc(i, j, k) + dydown(i, j - 1, k));

        dfdx = 0.25
               * (fddx_xhigh(i - 1, j, k) + fddx_xhigh(i, j, k)
                  + fddx_xhigh_down(i - 1, j - 1, k) + fddx_xhigh_down(i, j - 1, k));

        fout =
            0.25 * (ac(i, j, k) + adown(i, j - 1, k))
            * (
                // Component of flux from d/dz and d/dy
                (Jc(i, j, k) * g23c(i, j, k) + Jdown(i, j - 1, k) * g23down(i, j - 1, k))
                    * (dfdz - coef_yz * dfdy)
                // Non-orthogonal mesh correction with g12 metric
                + (Jc(i, j, k) * g12c(i, j, k)
                   + Jdown(i, j + 1, k) * g12down(i, j + 1, k))
                      * (dfdx - coef_xy * dfdy));

        yzresult(i, j, k) -= fout / (dyc(i, j, k) * Jc(i, j, k));
      }
    }
  }

  // Calculate the Y derivative, including in X guard cells
  // This is used to calculate X flux contribution from g12 * d/dy
  Field3D fddy(mesh);
  fddy.allocate();
  fddy.setDirectionY(yzresult.getDirectionY());
  for (int i = mesh->xstart - 1; i <= mesh->xend + 1; i++) {
    for (int j = mesh->ystart; j <= mesh->yend; j++) {
      for (int k = 0; k < mesh->LocalNz; k++) {
        fddy(i, j, k) =
            (fup(i, j + 1, k) - fdown(i, j - 1, k))
            / (0.5 * dyup(i, j + 1, k) + dyc(i, j, k) + 0.5 * dydown(i, j - 1, k));
      }
    }
  }

  // Z flux

  for (int i = mesh->xstart; i <= mesh->xend; i++) {
    for (int j = mesh->ystart; j <= mesh->yend; j++) {
      for (int k = 0; k < mesh->LocalNz; k++) {
        // Calculate flux between k and k+1
        int kp = (k + 1) % mesh->LocalNz;

        BoutReal ddx =
            0.5
            * ((fc(i + 1, j, k) - fc(i - 1, j, k))
                   / (0.5 * dxc(i + 1, j, k) + dxc(i, j, k) + 0.5 * dxc(i - 1, j, k))
               + (fc(i + 1, j, kp) - fc(i - 1, j, kp))
                     / (0.5 * dxc(i + 1, j, kp) + dxc(i, j, kp)
                        + 0.5 * dxc(i - 1, j, kp)));

        BoutReal ddy =
            0.5
            * ((fup(i, j + 1, k) - fdown(i, j - 1, k))
                   / (0.5 * dyup(i, j + 1, k) + dyc(i, j, k) + 0.5 * dydown(i, j - 1, k))
               + (fup(i, j + 1, kp) - fdown(i, j - 1, kp))
                     / (0.5 * dyup(i, j + 1, kp) + dyc(i, j, kp)
                        + 0.5 * dydown(i, j - 1, kp)));

        BoutReal ddz = 2. * (fc(i, j, kp) - fc(i, j, k)) / (dzc(i, j, k) + dzc(i, j, kp));

        BoutReal fout =
            0.5 * (ac(i, j, k) + ac(i, j, kp))
            * (
                // Jg^xz (d/dx - g_xy / g_yy d/dy)
                0.5
                    * (Jc(i, j, k) * coord->g13(i, j, k)
                       + Jc(i, j, kp) * coord->g13(i, j, kp))
                    * (ddx
                       - ddy * 0.5
                             * (g_12c(i, j, k) / SQ(Jc(i, j, k) * Bxyc(i, j, k))
                                + g_12c(i, j, kp) / SQ(Jc(i, j, kp) * Bxyc(i, j, kp))))
                // Jg^zz (d/dz - g_yz / g_yy d/dy)
                + 0.5
                      * (Jc(i, j, k) * coord->g33(i, j, k)
                         + Jc(i, j, kp) * coord->g33(i, j, kp))
                      * (ddz
                         - ddy * 0.5
                               * (g_23c(i, j, k) / SQ(Jc(i, j, k) * Bxyc(i, j, k))
                                  + g_23c(i, j, kp)
                                        / SQ(Jc(i, j, kp) * Bxyc(i, j, kp)))));

        yzresult(i, j, k) += fout / (Jc(i, j, k) * dzc(i, j, k));
        yzresult(i, j, kp) -= fout / (Jc(i, j, kp) * dzc(i, j, kp));
      }
    }
  }

  // Check if we need to transform back
  Field3D result;
  if (f.hasParallelSlices() && a.hasParallelSlices()) {
    result = yzresult;
  } else {
    result = fromFieldAligned(yzresult);
    fddy = fromFieldAligned(fddy);
  }

  // Flux in X

  for (int i = mesh->xstart - 1; i <= mesh->xend; i++) {
    for (int j = mesh->ystart; j <= mesh->yend; j++) {
      for (int k = 0; k < mesh->LocalNz; k++) {
        // Calculate flux from i to i+1

        const int kp = (k + 1) % mesh->LocalNz;
        const int km = (k - 1 + mesh->LocalNz) % mesh->LocalNz;

        BoutReal ddx = 2 * (f(i + 1, j, k) - f(i, j, k))
                       / (coord->dx(i, j, k) + coord->dx(i + 1, j, k));
        BoutReal ddy = 0.5 * (fddy(i, j, k) + fddy(i + 1, j, k));

        BoutReal ddz =
            0.5
            * ((f(i, j, kp) - f(i, j, km))
                   / (0.5 * coord->dz(i, j, kp) + coord->dz(i, j, k)
                      + 0.5 * coord->dz(i, j, km))
               + (f(i + 1, j, kp)
                  - f(i + 1, j, km)
                        / (0.5 * coord->dz(i + 1, j, kp) + coord->dz(i + 1, j, k)
                           + 0.5 * coord->dz(i + 1, j, km))));

        BoutReal fout =
            0.5 * (a(i, j, k) + a(i + 1, j, k))
            * (
                // Jg^xx (d/dx - g_xy / g_yy d/dy)
                0.5
                    * (coord->J(i, j, k) * coord->g11(i, j, k)
                       + coord->J(i + 1, j, k) * coord->g11(i + 1, j, k))
                    * (ddx
                       - ddy * 0.5
                             * (coord->g_12(i, j, k) / coord->g_22(i, j, k)
                                + coord->g_12(i + 1, j, k) / coord->g_22(i + 1, j, k)))
                // Jg^xz (d/dz - g_yz / g_yy d/dy)
                + 0.5
                      * ((coord->J(i, j, k) * coord->g13(i, j, k)
                          + coord->J(i + 1, j, k) * coord->g13(i + 1, j, k))
                         * (ddz
                            - ddy * 0.5
                                  * (coord->g_23(i, j, k) / coord->g_22(i, j, k)
                                     + coord->g_23(i + 1, j, k)
                                           / coord->g_22(i + 1, j, k)))));

        result(i, j, k) += fout / (coord->dx(i, j, k) * coord->J(i, j, k));
        result(i + 1, j, k) -= fout / (coord->dx(i + 1, j, k) * coord->J(i + 1, j, k));
      }
    }
  }

  return result;
}

CustomStencil::CustomStencil(Mesh &mesh, const std::string &name,
                             const std::string &type) {
  size_t num = 0;
  while (mesh.sourceHasVar(fmt::format("{:s}_{:s}_{:d}", name, type, num))) {
    num++;
  }
  ASSERT0(num > 1);
  coefs.resize(num);
  xoffset.resize(num);
  zoffset.resize(num);
  for (size_t i = 0; i < num; ++i) {
    Field3D ftmp(&mesh);
    mesh.get(ftmp, fmt::format("{:s}_{:s}_{:d}", name, type, i));
    coefs[i] = ftmp;
    int tmp;
    mesh.get(tmp, fmt::format("xoffset_{:s}_{:d}", type, i), 0);
    xoffset[i] = tmp;
    mesh.get(tmp, fmt::format("zoffset_{:s}_{:d}", type, i), 0);
    zoffset[i] = tmp;
  }
}

CustomStencil &CustomStencil::operator*=(BoutReal fac) {
  for (size_t i = 0; i < coefs.size(); ++i) {
    coefs[i] *= fac;
  }
  return *this;
}

Field3D CustomStencil::apply(const Field3D &a, const std::string &region) {
  auto result{zeroFrom(a)};
  BOUT_FOR(i, a.getRegion(region)) { result[i] = apply(a, i); }
  result.setRegion(region);
  return result;
}

BoutReal CustomStencil::apply(const Field3D &a, const Ind3D &i) {
  BoutReal result = 0;
  for (size_t c = 0; c < coefs.size(); ++c) {
    result += coefs[c][i] * a[i.xp(xoffset[c]).zpm(zoffset[c])];
  }
  return result;
}

namespace FCI {
Field3D dagp::operator()(const Field3D &a, const Field3D &f,
                         const std::string &region) {
  auto result{zeroFrom(f)};
  // auto aR = a * R;
  ASSERT1_FIELDS_COMPATIBLE(a, f);
  BOUT_FOR(i, f.getRegion(region)) {
    result[i] =
        ddR(a, i) * ddR(f, i) + ddZ(a, i) * ddZ(f, i) + a[i] * delp2(f, i);
  };
  result.name = fmt::format("dagp({:s}, {:s})", a.name, f.name);
  return result;
}

dagp::dagp(Mesh &mesh)
    : R(&mesh), ddR(mesh, "ddR"), ddZ(mesh, "ddZ"), delp2(mesh, "delp2") {
  mesh.get(R, "R");
};

dagp_fv::dagp_fv(Mesh &mesh)
    : fac_XX(&mesh), fac_XZ(&mesh), fac_ZX(&mesh), fac_ZZ(&mesh),
      volume(&mesh) {
  ASSERT0(mesh.get(fac_XX, "dagp_fv_XX") == 0);
  ASSERT0(mesh.get(fac_XZ, "dagp_fv_XZ") == 0);
  ASSERT0(mesh.get(fac_ZX, "dagp_fv_ZX") == 0);
  ASSERT0(mesh.get(fac_ZZ, "dagp_fv_ZZ") == 0);
  ASSERT0(mesh.get(volume, "dagp_fv_volume") == 0);
  mesh.addRegion("RGN_dapg_fv_xbndry",
                 Region<>(mesh.xstart - 1, mesh.xstart - 1, mesh.ystart,
                          mesh.yend, mesh.zstart, mesh.zend, mesh.LocalNy,
                          mesh.LocalNz));
}

Field3D dagp_fv::operator()(const Field3D &a, const Field3D &f) {
  auto result{zeroFrom(f)};
  ASSERT1_FIELDS_COMPATIBLE(a, f);
  BOUT_FOR(i, f.getRegion("RGN_dapg_fv_xbndry")) {
    result[i.xp()] = xflux(a, f, i);
  }
  BOUT_FOR(i, f.getRegion("RGN_NOBNDRY")) {
    const auto xf = xflux(a, f, i);
    result[i.xp()] += xf;
    result[i] -= xf;
    const auto zf = zflux(a, f, i);
    result[i.zp()] += zf;
    result[i] -= zf;
  }
  BOUT_FOR(i, f.getRegion("RGN_NOBNDRY")) { result[i] /= volume[i]; }
  return result;
}

inline BoutReal dagp_fv::xflux(const Field3D &a, const Field3D &f,
                               const Ind3D &i) {
  const auto ixp = i.xp();
  const auto av = 0.5 * (a[i] + a[ixp]);
  const auto dx = f[ixp] - f[i];
  const auto dz =
      0.5 * (f[i.zp()] - f[i.zm()] + f[i.zp().xp()] - f[i.zm().xp()]);
  return -(fac_XX[i] * dx + fac_XZ[i] * dz) * av;
}

inline BoutReal dagp_fv::zflux(const Field3D &a, const Field3D &f,
                               const Ind3D &i) {
  const auto izp = i.zp();
  const auto av = 0.5 * (a[i] + a[izp]);
  const auto dz = f[izp] - f[i];
  const auto dx = 0.5 * (f[i.xp()] - f[i.xm()] + f[izp.xp()] - f[izp.xm()]);
  return -(fac_ZX[i] * dx + fac_ZZ[i] * dz) * av;
}
} // namespace FCI
