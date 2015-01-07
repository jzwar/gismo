#pragma once

#include <iterator>

#include <gsCore/gsBasis.h>
#include <gsCore/gsGeometry.h>

#include <gsUtils/gsCombinatorics.h>

namespace gismo
{

template<class T>
gsMultiPatch<T>::gsMultiPatch(const gsGeometry<T> & geo )
    : gsBoxTopology( geo.parDim() )
{
    m_patches.push_back( geo.clone() );
    addBox();
    this->addAutoBoundaries();
}

template<class T>
gsMultiPatch<T>::gsMultiPatch( const gsMultiPatch& other )
    : gsBoxTopology( other ), m_patches( other.m_patches.size() )
{
    // clone all geometries
    cloneAll( other.m_patches.begin(), other.m_patches.end(),
              this->m_patches.begin() );
}

template<class T>
gsMultiPatch<T>::gsMultiPatch( const std::vector<gsGeometry<T> *>& patches )
    : gsBoxTopology( patches[0]->parDim(), patches.size() ) , m_patches( patches )
{
    this->addAutoBoundaries();
}

template<class T>
gsMultiPatch<T>::gsMultiPatch( const PatchContainer& patches,
                               const std::vector<patchSide>& boundary,
                               const std::vector<boundaryInterface>& interfaces )
    : gsBoxTopology( patches[0]->parDim(), patches.size(), boundary, interfaces ),
      m_patches( patches )
{ }

template<class T>
gsMultiPatch<T>::~gsMultiPatch()
{
    freeAll(m_patches);
}

template<class T>
int gsMultiPatch<T>::geoDim() const 
{
    GISMO_ASSERT( m_patches.size() > 0 , "Empty multipatch object.");
    return m_patches[0]->geoDim();
}

template<class T>
int gsMultiPatch<T>::coDim() const 
{
    GISMO_ASSERT( m_patches.size() > 0 , "Empty multipatch object.");
    return m_patches[0]->geoDim() - m_dim;
}

template<class T>
gsMatrix<T> 
gsMultiPatch<T>::parameterRange(int i) const
{
    return m_patches[i]->basis().support();
}

template<class T>
gsBasis<T> &
gsMultiPatch<T>::basis( std::size_t i ) const
{
    GISMO_ASSERT( i < m_patches.size(), "Invalid patch index requested from gsMultiPatch" );
    return m_patches[i]->basis();
}

template<class T>
std::vector<gsBasis<T> *> gsMultiPatch<T>::basesCopy() const
{
    std::vector<gsBasis<T> *> bb;
    for ( typename Base::const_iterator it = m_patches.begin();
          it != m_patches.end(); ++it ) {
        bb.push_back( ( *it )->basis().clone() );
    }
    return bb ;
}

template<class T>
void gsMultiPatch<T>::addPatch( gsGeometry<T>* g ) 
{
    if ( m_dim == -1 ) {
        m_dim = g->parDim();
    } else {
        assert( m_dim == g->parDim() );
    }
    m_patches.push_back( g ) ;
    addBox();
}

template<class T>
int gsMultiPatch<T>::findPatchIndex( gsGeometry<T>* g ) const {
    typename PatchContainer::const_iterator it
            = std::find( m_patches.begin(), m_patches.end(), g );
    assert( it != m_patches.end() );
    return it - m_patches.begin();
}

template<class T>
void gsMultiPatch<T>::addInterface( gsGeometry<T>* g1, boxSide s1,
                                    gsGeometry<T>* g2, boxSide s2 ) {
    int p1 = findPatchIndex( g1 );
    int p2 = findPatchIndex( g2 );
    gsBoxTopology::addInterface( p1, s1, p2, s2 );
}


template<class T>
void gsMultiPatch<T>::uniformRefine(int numKnots)
{
    for ( typename Base::const_iterator it = m_patches.begin();
          it != m_patches.end(); ++it )
    {
        ( *it )->uniformRefine(numKnots);
    }
}



/**
 * This is based on comparing the corners of the patch side and thus it implicitly assumes that that the patch faces match
 */
template<class T>
bool gsMultiPatch<T>::computeTopology( T tol )
{
    gsBoxTopology::clear();

    const size_t   np      = m_patches.size();
    const index_t  nCorP    = 1 << m_dim;     // corners per patch
    const index_t  nCorS    = 1 << (m_dim-1); // corners per side


    std::vector<gsMatrix<T> >  pCorners(np); // each matrix contains the physical coordinates of the vertexes of a patch
    std::vector<patchSide>     pSide;        // list of all patchSides to compare
    pSide.reserve(np * 2 * m_dim);
    for (size_t p=0; p<np; ++p)
    {
        // init the corners coordinated
        gsMatrix<T> supp  = m_patches[p]->parameterRange(); // the parameter domain of patch i
        gsMatrix<T> coor(m_dim,nCorP);                      // coordinates of the corners in parameter domain
        for (boxCorner c=boxCorner::getFirst(m_dim); c<boxCorner::getEnd(m_dim); ++c)
        {
            gsVector<bool> par   = c.parameters(m_dim);
            for (index_t i=0; i<m_dim;++i)
            {
                coor(i,c-1) = par(i) ? supp(i,1) : supp(i,0);
            }
        }
        m_patches[p]->eval_into(coor,pCorners[p]);
        // init the list of patchSides
        for (boxSide bs=boxSide::getFirst(m_dim); bs<boxSide::getEnd(m_dim); ++bs)
        {
            pSide.push_back(patchSide(p,bs));
        }
    }

    gsVector<index_t>      dirMap(m_dim);
    gsVector<bool>         dirO(m_dim);
    std::vector<boxCorner> cId1;
    std::vector<boxCorner> cId2;
    cId1.reserve(nCorS);
    cId2.reserve(nCorS);
    gsVector<bool>         matched;
    size_t other=0;
    while (pSide.size()>0)
    {
        bool done=false;
        const patchSide side = pSide.back();
        pSide.pop_back();
        for (other=0;other<pSide.size();++other)
        {
            side.getContainedCorners(m_dim,cId1);
            pSide[other].getContainedCorners(m_dim,cId2);

            matched.setConstant(cId2.size(),false);

            if ( matchVertecesOnSide( pCorners[side.patch], cId1, 0, pCorners[pSide[other].patch], cId2, matched, dirMap, dirO, tol ) )
            {
                dirMap(side.direction()) = pSide[other].direction();
                dirO(side.direction())   = !( side.parameter() == pSide[other].parameter() );
                gsBoxTopology::addInterface( boundaryInterface(side, pSide[other], dirMap, dirO));
                std::swap( pSide[other], pSide.back() );
                pSide.pop_back();
                done=true;
                break;
            }
        }
        if (!done)
            gsBoxTopology::addBoundary( side );
    }
    return true;
}


template <class T>
bool gsMultiPatch<T>::matchVertecesOnSide (
    const gsMatrix<T> &cc1, const std::vector<boxCorner> &ci1, index_t start,
    const gsMatrix<T> &cc2, const std::vector<boxCorner> &ci2, const gsVector<bool> &matched,
    gsVector<index_t> &dirMap, gsVector<bool>    &dirO,
    T tol,
    index_t reference)
{
    const bool computeOrientation = !(start&(start-1)) && (start != 0); // true if start is a power of 2
    const bool setReference       = start==0;          // if we search for the first point then we set the reference

    const int dim = cc1.rows();

    index_t o_dir=0;
    index_t d_dir=0;

    gsVector<bool> refPar;
    gsVector<bool> newPar;
    gsVector<bool> newMatched;

    if (computeOrientation)
    {
        // o_dir is the only direction in which the parameters differ between start and 0
        const gsVector<bool> parStart = ci1[start].parameters(dim);
        const gsVector<bool> parRef   = ci1[0].parameters(dim);
        for (; o_dir<dim && parStart(o_dir)==parRef(o_dir)  ;) ++o_dir;
    }
    if (!setReference)
    {
        refPar = ci2[reference].parameters(dim);
    }

    for (size_t j=0;j<ci2.size();++j)
    {
        if( !matched(j) && (cc1.col(ci1[start]-1)-cc2.col(ci2[j]-1)).norm() < tol )
        {
            index_t newRef =   (setReference) ? j : reference;
            if (computeOrientation)
            {
                index_t count=0;
                d_dir = 0;
                newPar =  ci2[j].parameters(dim);
                for (index_t i=0; i< newPar.rows();++i)
                {
                    if ( newPar(i)!=refPar(i) )
                    {
                        d_dir=i;
                        ++count;
                    }
                }
                if (count != 1)
                {
                    // the match is wrong: we are mapping an edge to a diagonal
                    continue;
                }
                dirMap(o_dir) = d_dir;
                dirO  (o_dir) = (static_cast<index_t>(j)>reference);
            }
            if ( start + 1 == index_t( ci1.size() ) )
            {
                // we matched the last vertex, we are done
                return true;
            }
            newMatched = matched;
            newMatched(j) = true;
            if (!matchVertecesOnSide( cc1,ci1,start+1,cc2,ci2,newMatched,dirMap,dirO, tol,newRef))
                continue;
            else
                return true;
        }
    }
    return false;
}


template<class T>
gsAffineFunction<T> gsMultiPatch<T>::getMapForInterface(const boundaryInterface &bi, T scaling) const
{
    if (scaling==0)
        scaling=1;
    gsMatrix<T> box1=m_patches[bi.first().patch]->support();
    gsMatrix<T> box2=m_patches[bi.second().patch]->support();
    const index_t oDir1 = bi.first().direction();
    const index_t oDir2 = bi.second().direction();
    const T len1=box1(oDir1,1)-box1(oDir1,0);
    if (bi.second().parameter())
    {
        box2(oDir2,0)=box2(oDir2,1);
        box2(oDir2,1)+=scaling*len1;
    }
    else
    {
        box2(oDir2,1)=box2(oDir2,0);
        box2(oDir2,0)-=scaling*len1;
    }
    return gsAffineFunction<T>(bi.dirMap(bi.first()),bi.dirOrientation(bi.first()) ,box1,box2);
}

}
