#ifndef JERI_TEI_H
#define JERI_TEI_H

#include <libint2.hpp>
#include <jlcxx/jlcxx.hpp>
//#include <jlcxx/stl.hpp>

//#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <limits>
#include <vector>

typedef int64_t julia_int;

//------------------------------------------------------------------------//
//-- C++ JERI engine: small wrapper allowing LibInt to be used in Julia --//
//-- Two Electron Integral Engine: Wrapper class for calls to ------------//
//-- libint2::Engine m_coulomb_eng.compute2 ------------------------------//
//-- Base class defining the interface for Two Electron integral engines--//
//------------------------------------------------------------------------//

class TEIEngine
{
public:
  const libint2::BasisSet *m_basis_set;
  const libint2::ShellPair *m_shellpair_data;

  libint2::Engine m_coulomb_eng;
  TEIEngine(const libint2::BasisSet &t_basis_set,
            const std::vector<libint2::ShellPair> &t_shellpair_data, const int max_nprim, const int max_l)
      : m_basis_set(&t_basis_set),
        m_shellpair_data(t_shellpair_data.data()),
        m_coulomb_eng(libint2::Operator::coulomb,
                      max_nprim,
                      max_l,
                      0)
  {
    m_coulomb_eng.set_precision(0.0); // no screening in the c++ engine
  };

  ~TEIEngine(){};
  virtual bool compute_eri_block(jlcxx::ArrayRef<double> eri_block,
                                 julia_int ash, julia_int bsh, julia_int csh, julia_int dsh,
                                 julia_int bra_idx, julia_int ket_idx,
                                 julia_int absize, julia_int cdsize){};
};

//------------------------------------------------------------------------//
//-- Two electron integral engine for Restricted Hartree Fock ------------//
//------------------------------------------------------------------------//

class RHFTEIEngine : public TEIEngine
{

public:
  //-- ctors and dtors --//
  RHFTEIEngine(const libint2::BasisSet &t_basis_set,
               const std::vector<libint2::ShellPair> &t_shellpair_data)
      : TEIEngine(t_basis_set, t_shellpair_data, (int)t_basis_set.max_nprim(), (int)t_basis_set.max_l()){};

  ~RHFTEIEngine(){};

  //-- member functions --//
  inline bool compute_eri_block(jlcxx::ArrayRef<double> eri_block,
                                julia_int ash, julia_int bsh, julia_int csh, julia_int dsh,
                                julia_int bra_idx, julia_int ket_idx,
                                julia_int absize, julia_int cdsize)
  {
    m_coulomb_eng.compute2<libint2::Operator::coulomb,
                           libint2::BraKet::xx_xx, 0>((*m_basis_set)[ash - 1], (*m_basis_set)[bsh - 1],
                                                      (*m_basis_set)[csh - 1], (*m_basis_set)[dsh - 1],
                                                      &m_shellpair_data[bra_idx - 1], &m_shellpair_data[ket_idx - 1]);

    //assert(m_coulomb_eng.results()[0] != nullptr);
    if (m_coulomb_eng.results()[0] != nullptr)
    {
      memcpy(eri_block.data(), m_coulomb_eng.results()[0],
             absize * cdsize * sizeof(double));

      return false;
    }
    else
    {
      return true;
    }
  }
};

//------------------------------------------------------------------------//
//-- Two electron integral engine for Density Fitted ---------------------//
//-- Restricted Hartree Fock ---------------------------------------------//
//------------------------------------------------------------------------//

class DFRHFTEIEngine : public TEIEngine
{

public:
  const libint2::BasisSet *m_auxillary_basis_set;
  const libint2::ShellPair *m_auxillary_shellpair_data;
  libint2::Engine m_two_center_engine;

  //-- ctors and dtors --//
  DFRHFTEIEngine(
      const libint2::BasisSet &t_basis_set,
      const libint2::BasisSet &t_auxillary_basis_set,
      const std::vector<libint2::ShellPair> &t_shellpair_data,
      const std::vector<libint2::ShellPair> &t_auxillary_shellpair_data)
      : TEIEngine(t_basis_set, t_shellpair_data, (int)std::max(t_basis_set.max_nprim(), t_auxillary_basis_set.max_nprim()),
                  (int)std::max(t_basis_set.max_l(), t_auxillary_basis_set.max_l()))
  {
    m_two_center_engine = libint2::Engine(libint2::Operator::coulomb,
                                          t_basis_set.max_nprim(), t_basis_set.max_l(), 0);
    m_auxillary_basis_set = &t_auxillary_basis_set;
    m_auxillary_shellpair_data = t_auxillary_shellpair_data.data();
  }

  ~DFRHFTEIEngine(){};

  //-- member functions --//

  //-- compute_eri_block --//
  //-- ash, bsh, csh, dsh, are shell indicies --//
  inline bool compute_eri_block(julia_int ash, julia_int bsh, julia_int csh, julia_int dsh,
                                julia_int bra_idx, julia_int ket_idx,
                                julia_int absize, julia_int cdsize)
  {
    auto unitShell = libint2::Shell::unit();
    m_coulomb_eng.compute2<libint2::Operator::coulomb,
                           libint2::BraKet::xs_xx, 0>((*m_auxillary_basis_set)[ash - 1], unitShell,
                                                      (*m_basis_set)[csh - 1], (*m_basis_set)[dsh - 1]);
    std::cout << "(dfbs[" << ash - 1 << "], unitshell||obs[" << csh - 1 << "], obs[" << dsh - 1 << "]) = " << *m_coulomb_eng.results()[0] << std::endl;
    if (m_coulomb_eng.results()[0] != nullptr)
    {
      return false;

      // memcpy(eri_block.data(), m_coulomb_eng.results()[0],
      //        absize * cdsize * sizeof(double));

      // return false;
    }
    else
    {
      //memset(eri_block.data(), 0.0, absize*cdsize*sizeof(double));
      return true;
    }
  }
  inline bool compute_two_center_eri_block(julia_int ash, julia_int bsh)
  {
    m_two_center_engine.compute((*m_auxillary_basis_set)[ash - 1], (*m_auxillary_basis_set)[bsh - 1]);
    std::cout << "(obs[" << ash - 1 << "]||obs[" << bsh - 1 << "]) = " << *m_two_center_engine.results()[0] << std::endl;
    return false;
  }
};

#endif /* JERI_TEI_H */
