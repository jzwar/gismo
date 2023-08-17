#=============================================================================
# Handling of X86 / X86_64 options
#
# This is a three-step process:
#
# 1. Generate a list of available compiler flags for the specific CPU
#
# 2. Enable/disable feature flags based on available CPU features,
#    used-defined USE_<feature> variables and the capabilities of the
#    host system's compiler and linker
#
# 3. Set compiler-specific flags (e.g., -m<feature>/-mno-<feature>)
#=============================================================================

include(ofa/CommonMacros)

macro(OFA_HandleX86Options)

  # Special treatment for "native" flag
  if(TARGET_ARCHITECTURE STREQUAL "native")
    if(MSVC)
      # MSVC (on Windows)
      message(FATAL_ERROR "[OFA] MSVC does not support \"native\" flag.")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel"
        OR CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
      if(WIN32)
        # Intel (on Windows)
        AddCXXCompilerFlag("/QxHOST" FLAGS ARCHITECTURE_CXX_FLAGS RESULT _ok)
      else()
        # Intel (on Linux)
        AddCXXCompilerFlag("-xHOST" FLAGS ARCHITECTURE_CXX_FLAGS RESULT _ok)
      endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC"
        OR CMAKE_CXX_COMPILER_ID MATCHES "PGI")
      # NVidia HPC / PGI (on Linux/Windows)
      AddCXXCompilerFlag("-tp=native" FLAGS ARCHITECTURE_CXX_FLAGS RESULT _ok)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "SunPro")
      # Sun/Oracle Studio (on Linux/Sun OS)
      AddCXXCompilerFlag("-native" FLAGS ARCHITECTURE_CXX_FLAGS RESULT _ok)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Cray")
      # Cray (on Linux)
      message(FATAL_ERROR, "[OFA] Cray compiler does not support \"native\" flag.")
    else()
      # Others: GNU, Clang and variants
      AddCXXCompilerFlag("-march=native" FLAGS ARCHITECTURE_CXX_FLAGS RESULT _ok)
    endif()

    if(NOT _ok)
      message(FATAL_ERROR "[OFA] An error occured while setting the \"native\" flag.")
    endif()

  elseif(NOT TARGET_ARCHITECTURE STREQUAL "none")

    # Step 1: Generate a list of compiler flags for the specific CPU
    set(_march_flag_list)
    set(_available_extension_list)

    # Define macros for Intel
    macro(_nehalem)
      list(APPEND _march_flag_list "nehalem")
      list(APPEND _march_flag_list "corei7")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_extension_list "mmx" "sse" "sse2" "sse3" "ssse3" "sse4.1" "sse4.2" "popcnt")
    endmacro()
    macro(_westmere)
      list(APPEND _march_flag_list "westmere")
      _nehalem()
      list(APPEND _available_extension_list "aes" "pclmul")
    endmacro()
    macro(_sandybridge)
      list(APPEND _march_flag_list "sandybridge")
      list(APPEND _march_flag_list "corei7-avx")
      _westmere()
      list(APPEND _available_extension_list "avx")
    endmacro()
    macro(_ivybridge)
      list(APPEND _march_flag_list "ivybridge")
      list(APPEND _march_flag_list "core-avx-i")
      _sandybridge()
      list(APPEND _available_extension_list "rdrnd" "f16c" "fsgsbase")
    endmacro()
    macro(_haswell)
      list(APPEND _march_flag_list "haswell")
      list(APPEND _march_flag_list "core-avx2")
      _ivybridge()
      list(APPEND _available_extension_list "abm" "avx2" "fma" "bmi" "bmi2")
    endmacro()
    macro(_broadwell)
      list(APPEND _march_flag_list "broadwell")
      _haswell()
      list(APPEND _available_extension_list "rdseed" "adx" "prfchw")
    endmacro()
    macro(_skylake)
      list(APPEND _march_flag_list "skylake")
      _broadwell()
      list(APPEND _available_extension_list "clflushopt" "xsavec" "xsaves")
    endmacro()
    macro(_skylake_avx512)
      list(APPEND _march_flag_list "skylake-avx512")
      _skylake()
      list(APPEND _available_extension_list "avx512bw" "avx512cd" "avx512dq" "avx512f" "avx512vl" "clwb" "pku")
    endmacro()
    macro(_cascadelake)
      list(APPEND _march_flag_list "cascadelake")
      _skylake_avx512()
      list(APPEND _available_extension_list "avx512vnni")
    endmacro()
    macro(_cooperlake)
      list(APPEND _march_flag_list "cooperlake")
      _cascadelake()
      list(APPEND _available_extension_list "avx512bf16")
    endmacro()
    macro(_cannonlake)
      list(APPEND _march_flag_list "cannonlake")
      _skylake()
      list(APPEND _available_extension_list "avx512bw" "avx512cd" "avx512dq" "avx512f" "avx512vl" "clwb" "pku" "avx512ifma" "avx512vbmi" "sha" "umip")
    endmacro()
    macro(_icelake)
      list(APPEND _march_flag_list "icelake-client")
      _cannonlake()
      list(APPEND _available_extension_list "avx512bitalg" "avx512vbmi2" "avx512vnni" "avx512vpopcntdq" "clwb" "gfni" "rdpid" "vaes" "vpclmulqdq")
    endmacro()
    macro(_icelake_avx512)
      list(APPEND _march_flag_list "icelake-server")
      _icelake()
      list(APPEND _available_extension_list "pconfig" "wbnoinvd")
    endmacro()
    macro(_tigerlake)
      list(APPEND _march_flag_list "tigerlake")
      _icelake()
      list(APPEND _available_extension_list "avx512vp2intersect" "keylocker" "movdir64b" "movdiri" "pconfig" "wbnoinvd")
    endmacro()
    macro(_alderlake)
      list(APPEND _march_flag_list "alderlake")
      _broadwell()
      list(APPEND _available_extension_list "avxvnni" "cldemote" "clwb" "gfni" "hreset" "kl" "lzcnt" "movdir64b" "movdiri" "pconfig" "pku" "ptwrite" "rdpid" "serialize" "sgx" "umip" "vaes" "vpclmulqdq" "waitpkg" "widekl" "xsave" "xsavec" "xsaveopt" "xsaves")
    endmacro()
    macro(_sapphirerapids)
      list(APPEND _march_flag_list "sapphirerapids")
      _skylake_avx512()
      list(APPEND _available_extension_list "amx-bf16" "amx-int8" "amx-tile" "avxvnni" "avx512bf16" "avx512vnni" "avx512vp2intersect" "cldemote" "enqcmd" "movdir64b" "movdiri" "ptwrite" "serialize" "tsxldtrk" "uintr" "waitpkg")
    endmacro()
    macro(_rocketlake)
      list(APPEND _march_flag_list "rocketlake")
      _skylake_avx512()
      list(APPEND _available_extension_list "avx512bitalg" "avx512ifma" "avx512vbmi" "avx512vbmi2" "avx512vnni" "avx512vpopcntdq" "gfni" "rdpid" "sha" "umip" "vaes" "vpclmulqdq")
    endmacro()
    macro(_raptorlake)
      list(APPEND _march_flag_list "raptorlake")
      _skylake_avx512()
      list(APPEND _available_extension_list "avx512bitalg" "avx512ifma" "avx512vbmi" "avx512vbmi2" "avx512vnni" "avx512vpopcntdq" "gfni" "rdpid" "sha" "umip" "vaes" "vpclmulqdq")
    endmacro()
    macro(_knightslanding)
      list(APPEND _march_flag_list "knl")
      _broadwell()
      list(APPEND _available_extension_list "avx512f" "avx512pf" "avx512er" "avx512cd")
    endmacro()
    macro(_knightsmill)
      list(APPEND _march_flag_list "knm")
      _broadwell()
      list(APPEND _available_extension_list "avx512f" "avx512pf" "avx512er" "avx512cd" "avx5124fmaps" "avx5124vnni" "avx512vpopcntdq")
    endmacro()
    macro(_silvermont)
      list(APPEND _march_flag_list "silvermont")
      _westmere()
      list(APPEND _available_extension_list "rdrnd")
    endmacro()
    macro(_goldmont)
      list(APPEND _march_flag_list "goldmont")
      _silvermont()
      list(APPEND _available_extension_list "rdseed")
    endmacro()
    macro(_goldmont_plus)
      list(APPEND _march_flag_list "goldmont-plus")
      _goldmont()
      list(APPEND _available_extension_list "rdpid")
    endmacro()
    macro(_tremont)
      list(APPEND _march_flag_list "tremont")
      _goldmont_plus()
    endmacro()

    # Define macros for AMD
    macro(_k8)
      list(APPEND _march_flag_list "k8")
      list(APPEND _available_extension_list "mmx" "3dnow" "sse" "sse2")
    endmacro()
    macro(_k8_sse3)
      list(APPEND _march_flag_list "k8-sse3")
      _k8()
      list(APPEND _available_extension_list "sse3")
    endmacro()
    macro(_barcelona) # amd10h
      list(APPEND _march_flag_list "barcelona")
      _k8_sse3()
      list(APPEND _available_extension_list "sse4a" "abm")
    endmacro()
    macro(_amd14h)
      list(APPEND _march_flag_list "btver1")
      _barcelona()
      list(APPEND _available_extension_list "cx16" "ssse3")
    endmacro()
    macro(_bulldozer) # amd15h
      list(APPEND _march_flag_list "bdver1")
      _amd14h()
      list(APPEND _available_extension_list "sse4.1" "sse4.2" "avx" "xop" "fma4" "lwp" "aes" "pclmul")
    endmacro()
    macro(_piledriver)
      list(APPEND _march_flag_list "bdver2")
      _bulldozer()
      list(APPEND _available_extension_list "fma" "f16c" "bmi" "tbm")
    endmacro()
    macro(_steamroller)
      list(APPEND _march_flag_list "bdver3")
      _piledriver()
      list(APPEND _available_extension_list "fsgsbase")
    endmacro()
    macro(_excavator)
      list(APPEND _march_flag_list "bdver4")
      _steamroller()
      list(APPEND _available_extension_list "bmi2" "avx2" "movbe")
    endmacro()
    macro(_amd16h)
      list(APPEND _march_flag_list "btver2")
      _amd14h()
      list(APPEND _available_extension_list "movbe" "sse4.1" "sse4.2" "avx" "f16c" "bmi" "pclmul" "aes")
    endmacro()
    macro(_zen)
      list(APPEND _march_flag_list "znver1")
      _amd16h()
      list(APPEND _available_extension_list "bmi2" "fma" "fsgsbase" "avx2" "adcx" "rdseed" "mwaitx" "sha" "clzero" "xsavec" "xsaves" "clflushopt" "popcnt")
    endmacro()
    macro(_zen2)
      list(APPEND _march_flag_list "znver2")
      _zen()
      list(APPEND _available_extension_list "clwb" "rdpid" "wbnoinvd")
    endmacro()
    macro(_zen3)
      list(APPEND _march_flag_list "znver3")
      _zen3()
      list(APPEND _available_extension_list "pku" "vpclmulqdq" "vaes")
    endmacro()

    # Intel
    if(TARGET_ARCHITECTURE STREQUAL "core" OR TARGET_ARCHITECTURE STREQUAL "core2")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_extension_list "mmx" "sse" "sse2" "sse3")
    elseif(TARGET_ARCHITECTURE STREQUAL "merom")
      list(APPEND _march_flag_list "merom")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_extension_list "mmx" "sse" "sse2" "sse3" "ssse3")
    elseif(TARGET_ARCHITECTURE STREQUAL "penryn")
      list(APPEND _march_flag_list "penryn")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_extension_list "mmx" "sse" "sse2" "sse3" "ssse3")
      message(STATUS "[OFA] Sadly the Penryn architecture exists in variants with SSE4.1 and without SSE4.1.")
      if(_cpu_flags MATCHES "sse4_1")
        message(STATUS "[OFA] SSE4.1: enabled (auto-detected from this computer's CPU flags)")
        list(APPEND _available_extension_list "sse4.1")
      else()
        message(STATUS "[OFA] SSE4.1: disabled (auto-detected from this computer's CPU flags)")
      endif()
    elseif(TARGET_ARCHITECTURE STREQUAL "knm")
      _knightsmill()
    elseif(TARGET_ARCHITECTURE STREQUAL "knl")
      _knightslanding()
    elseif(TARGET_ARCHITECTURE STREQUAL "raptorlake")
      _raptorlake()
    elseif(TARGET_ARCHITECTURE STREQUAL "rocketlake")
      _rocketlake()
    elseif(TARGET_ARCHITECTURE STREQUAL "sapphirerapids")
      _sapphirerapids()
    elseif(TARGET_ARCHITECTURE STREQUAL "alderlake")
      _alderlake()
    elseif(TARGET_ARCHITECTURE STREQUAL "tigerlake")
      _tigerlake()
    elseif(TARGET_ARCHITECTURE STREQUAL "icelake")
      _icelake()
    elseif(TARGET_ARCHITECTURE STREQUAL "icelake-xeon" OR TARGET_ARCHITECTURE STREQUAL "icelake-avx512")
      _icelake_avx512()
    elseif(TARGET_ARCHITECTURE STREQUAL "cannonlake")
      _cannonlake()
    elseif(TARGET_ARCHITECTURE STREQUAL "cooperlake")
      _cooperlake()
    elseif(TARGET_ARCHITECTURE STREQUAL "cascadelake")
      _cascadelake()
    elseif(TARGET_ARCHITECTURE STREQUAL "kabylake")
      _skylake()
    elseif(TARGET_ARCHITECTURE STREQUAL "skylake-xeon" OR TARGET_ARCHITECTURE STREQUAL "skylake-avx512")
      _skylake_avx512()
    elseif(TARGET_ARCHITECTURE STREQUAL "skylake")
      _skylake()
    elseif(TARGET_ARCHITECTURE STREQUAL "broadwell")
      _broadwell()
    elseif(TARGET_ARCHITECTURE STREQUAL "haswell")
      _haswell()
    elseif(TARGET_ARCHITECTURE STREQUAL "ivybridge")
      _ivybridge()
    elseif(TARGET_ARCHITECTURE STREQUAL "sandybridge")
      _sandybridge()
    elseif(TARGET_ARCHITECTURE STREQUAL "westmere")
      _westmere()
    elseif(TARGET_ARCHITECTURE STREQUAL "nehalem")
      _nehalem()
    elseif(TARGET_ARCHITECTURE STREQUAL "tremont")
      _tremont()
    elseif(TARGET_ARCHITECTURE STREQUAL "goldmont-plus")
      _goldmont_plus()
    elseif(TARGET_ARCHITECTURE STREQUAL "goldmont")
      _goldmont()
    elseif(TARGET_ARCHITECTURE STREQUAL "silvermont")
      _silvermont()
    elseif(TARGET_ARCHITECTURE STREQUAL "bonnell")
      list(APPEND _march_flag_list "bonnell")
      list(APPEND _march_flag_list "atom")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_extension_list "sse" "sse2" "sse3" "ssse3")
    elseif(TARGET_ARCHITECTURE STREQUAL "atom")
      list(APPEND _march_flag_list "atom")
      list(APPEND _march_flag_list "core2")
      list(APPEND _available_extension_list "sse" "sse2" "sse3" "ssse3")

    # AMD
    elseif(TARGET_ARCHITECTURE STREQUAL "k8")
      _k8()
    elseif(TARGET_ARCHITECTURE STREQUAL "k8-sse3")
      k8_sse3()
    elseif(TARGET_ARCHITECTURE STREQUAL "barcelona" OR
           TARGET_ARCHITECTURE STREQUAL "istanbul" OR
           TARGET_ARCHITECTURE STREQUAL "magny-cours")
      _barcelona()
    elseif(TARGET_ARCHITECTURE STREQUAL "amd14h")
      _amd14h()
    elseif(TARGET_ARCHITECTURE STREQUAL "bulldozer" OR
           TARGET_ARCHITECTURE STREQUAL "interlagos")
      _bulldozer()
    elseif(TARGET_ARCHITECTURE STREQUAL "piledriver")
      _piledriver()
    elseif(TARGET_ARCHITECTURE STREQUAL "steamroller")
      _steamroller()
    elseif(TARGET_ARCHITECTURE STREQUAL "excavator")
      _excavator()
    elseif(TARGET_ARCHITECTURE STREQUAL "amd16h")
      _amd16h()
    elseif(TARGET_ARCHITECTURE STREQUAL "zen")
      _zen()
    elseif(TARGET_ARCHITECTURE STREQUAL "zen2")
      _zen2()
    elseif(TARGET_ARCHITECTURE STREQUAL "zen3")
      _zen3()

      # Others
    elseif(TARGET_ARCHITECTURE STREQUAL "generic")
      list(APPEND _march_flag_list "generic")
      list(APPEND _available_extension_list "sse")
    elseif(TARGET_ARCHITECTURE STREQUAL "none")
      # add this clause to remove it from the else clause

    else()
      message(FATAL_ERROR "[OFA] Unknown target architecture: \"${TARGET_ARCHITECTURE}\". Please set TARGET_ARCHITECTURE to a supported value.")
    endif()

    # Clean list of available extensions
    list(SORT _available_extension_list)
    list(REMOVE_DUPLICATES _available_extension_list)

    if(OFA_VERBOSE)
      if(_march_flag_list)
        string(REPLACE ";"  ", " _str "${_march_flag_list}")
        string(TOUPPER ${_str} _str)
        message(STATUS "[OFA] CPU architectures: " ${_str})
      endif()
      if(_available_extension_list)
        list(LENGTH _available_extension_list _len)
        string(REPLACE ";"  ", " _str "${_available_extension_list}")
        string(TOUPPER ${_str} _str)
        message(STATUS "[OFA] Extensions (${_len} available): ${_str}")
      endif()
    endif()

    set(_check_extension_list)
    set(_check_extension_flag_list)
    set(_disable_extension_flag_list)
    set(_enable_extension_flag_list)
    set(_ignore_extension_flag_list)

    # Set compiler-specific option names
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
      set(_enable_flag "/arch:")
      unset(_disable)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "SunPro")
      set(_enable_flag "-xarch=")
      unset(_disable_flag)
    else()
      set(_enable_flag "-m")
      set(_disable_flag "-mno-")
    endif()

    # Step 2: Enable/disable feature flags based on available CPU
    #         features, used-defined USE_<feature> variables and
    #         the capabilities of the host system's compiler and linker
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ofa/ChecksX86.txt _checks)
    string(REGEX REPLACE "[:;]" "|" _checks "${_checks}")
    string(REPLACE "\n" ";" _checks "${_checks}")

    set(_skip_check FALSE)

    # Iterate over the list of checks line by line
    foreach (_check ${_checks})
      string(REPLACE "|" ";" _check "${_check}")

      # Parse for special lines
      if ("${_check}" MATCHES "^#" ) # Skip comment
        continue()

      elseif ("${_check}" MATCHES "^push_enable" ) # Start enable block
        list(GET _check 1 _push_enable_list)
        string(REPLACE "," ";" _push_enable_list "${_push_enable_list}")
        _ofa_find(_push_enable_list "${CMAKE_CXX_COMPILER_ID}" _found)
        if(_found)
          list(INSERT _skip_check 0 FALSE)
        else()
          list(INSERT _skip_check 0 TRUE)
        endif()
        continue()

      elseif ("${_check}" MATCHES "^pop_enable" ) # End enable block
        list(REMOVE_AT _skip_check 0)
        continue()

      elseif ("${_check}" MATCHES "^push_disable" ) # Start disable block
        list(GET _check 1 _push_disable_list)
        string(REPLACE "," ";" _push_disable_list "${_push_disable_list}")
        _ofa_find(_push_disable_list "${CMAKE_CXX_COMPILER_ID}" _found)
        if(_found)
          list(INSERT _skip_check 0 TRUE)
        else()
          # Compiler was not found in the list, so we keep its previous status
          list(GET _skip_check 0 _skip)
          list(INSERT _skip_check 0 ${_skip})
        endif()
        continue()

      elseif ("${_check}" MATCHES "^pop_disable" ) # End disable block
        list(REMOVE_AT _skip_check 0)
        continue()
      endif()

      # Skip test?
      list(GET _skip_check 0 _skip)
      if(_skip)
        continue()
      endif()

      # Extract extra CPU extensions, header files, function name, and parameters
      list(GET _check 0 _check_extension_flags)
      list(GET _check 1 _check_headers)
      list(GET _check 2 _check_function)
      list(GET _check 3 _check_params)

      # Convert list of extensions into compiler flags
      string(REPLACE "," ";" _check_extension_flags "${_check_extension_flags}")
      list(GET _check_extension_flags 0 _extension_flag)
      list(APPEND _check_extension_flag_list "${_extension_flag}")
      string(REPLACE ";" " ${_enable_flag}" _check_extra_flags " ${_enable_flag}${_check_extension_flags}")

      # Extract optional extension alias
      list(LENGTH _check _len)
      if(${_len} EQUAL 5)
        list(GET _check 4 _extension)
      else()
        set(_extension "${_extension_flag}")
      endif()

      list(APPEND _check_extension_list "${_extension}")

      # Define USE_<_extension_flag> variable
      set(_useVar "USE_${_extension_flag}")
      string(TOUPPER "${_useVar}" _useVar)
      string(REPLACE "[-.+/:= ]" "_" _useVar "${_useVar}")

      # If not specified externally, set the value of the
      # USE_<_extension_flag> variable to TRUE if it is found in the list
      # of available extensions and FALSE otherwise
      if(NOT DEFINED ${_useVar})
        _ofa_find(_available_extension_list "${_extension}" _found)
        set(${_useVar} ${_found})
      endif()

      if(${_useVar})
        # Check if the compiler supports the -m<_extension_flag>
        # flag and can compile the provided test code with it
        set(_code "\nint main() { ${_check_function}(${_check_params})\; return 0\; }")
        AddCXXCompilerFlag("${_enable_flag}${_extension_flag}"
          EXTRA_FLAGS ${_check_extra_flags}
          HEADERS     ${_check_headers}
          CODE        "${_code}"
          RESULT      _ok)
        if(NOT ${_ok})
          # Test failed
          set(${_useVar} FALSE CACHE BOOL "Use ${_extension} extension.")
        else()
          # Test succeeded
          set(${_useVar} TRUE CACHE BOOL "Use ${_extension} extension.")
        endif()
      else()
        # Disable extension without running tests
        set(${_useVar} FALSE CACHE BOOL "Use ${_extension} extension.")
      endif()
      mark_as_advanced(${_useVar})
    endforeach()

    # Generate lists of enabled/disabled flags
    list(REMOVE_DUPLICATES _check_extension_flag_list)
    foreach(_extension_flag ${_check_extension_flag_list})
      _ofa_find(_available_extension_list "${_extension_flag}" _found)
      set(_useVar "USE_${_extension_flag}")
      string(TOUPPER "${_useVar}" _useVar)
      string(REPLACE "[-.+/:= ]" "_" _useVar "${_useVar}")

      if(${_useVar})
        # Add <_extension_flag> to list of enabled extensions (if supported)
        set(_haveVar "HAVE_${_enable_flag}${_extension_flag}")
        string(REGEX REPLACE "[-.+/:= ]" "_" _haveVar "${_haveVar}")
        if(NOT ${_haveVar})
          if(OFA_VERBOSE)
            message(STATUS "[OFA] Ignoring flag ${_enable_flag}${_extension_flag} because checks failed")
          endif()
          list(APPEND _ignore_extension_flag_list "${_extension_flag}")
          continue()
        endif()
        list(APPEND _enable_extension_flag_list "${_extension_flag}")
      elseif(DEFINED _disable_flag)
        # Add <_extension_flag> to list of disabled extensions (if supported)
        AddCXXCompilerFlag("${_disable_flag}${_extension_flag}")
        set(_haveVar "HAVE_${_disable_flag}${_extension_flag}")
        string(REGEX REPLACE "[-.+/:= ]" "_" _haveVar "${_haveVar}")
        if(NOT ${_haveVar})
          if(OFA_VERBOSE)
            message(STATUS "[OFA] Ignoring flag ${_disable_flag}${_extension_flag} because checks failed")
          endif()
          list(APPEND _ignore_extension_flag_list "${_extension_flag}")
          continue()
        endif()
        list(APPEND _disable_extension_flag_list "${_extension_flag}")
      else()
        list(APPEND _ignore_extension_flag_list "${_extension_flag}")
      endif()
    endforeach()

    if(OFA_VERBOSE)
      # Print checked extension flags
      if(_check_extension_flag_list)
        list(LENGTH _check_extension_flag_list _len)
        list(SORT _check_extension_flag_list)
        string(REPLACE ";"  ", " _str "${_check_extension_flag_list}")
        string(TOUPPER ${_str} _str)
        message(STATUS "[OFA] Extensions (${_len} checked): ${_str}")
      endif()
      # Print enabled extension flags
      if(_enable_extension_flag_list)
        list(LENGTH _enable_extension_flag_list _len)
        list(SORT _enable_extension_flag_list)
        string(REPLACE ";"  ", " _str "${_enable_extension_flag_list}")
        string(TOUPPER ${_str} _str)
        message(STATUS "[OFA] Extensions (${_len} enabled): ${_str}")
      endif()
      # Print disabled extension flags
      if(_disable_extension_flag_list)
        list(LENGTH _disable_extension_flag_list _len)
        list(SORT _disable_extension_flag_list)
        string(REPLACE ";"  ", " _str "${_disable_extension_flag_list}")
        string(TOUPPER ${_str} _str)
        message(STATUS "[OFA] Extensions (${_len} disabled): ${_str}")
      endif()
      # Print ignored extension flags
      if(_ignore_extension_flag_list)
        list(LENGTH _ignore_extension_flag_list _len)
        list(SORT _ignore_extension_flag_list)
        string(REPLACE ";"  ", " _str "${_ignore_extension_flag_list}")
        string(TOUPPER ${_str} _str)
        message(STATUS "[OFA] Extensions (${_len} ignored): ${_str}")
      endif()
      # Print unhandled extension flags
      set(_unhandled_extension_list)
      foreach(_extension ${_available_extension_list})
        _ofa_find(_check_extension_list "${_extension}" _found)
        if(NOT _found)
          list(APPEND _unhandled_extension_list ${_extension})
        endif()
      endforeach()
      if(_unhandled_extension_list)
        list(LENGTH _unhandled_extension_list _len)
        list(SORT _unhandled_extension_list)
        string(REPLACE ";"  ", " _str "${_unhandled_extension_list}")
        string(TOUPPER ${_str} _str)
        message(STATUS "[OFA] Extensions (${_len} unhandled): ${_str}")
      endif()
    endif()

    # Step 3: Set compiler-specific flags (e.g., -m<feature>/-mno-<feature>)
    if(MSVC AND MSVC_VERSION GREATER 1700)
      _ofa_find(_enable_extension_flag_list "avx512f" _found)
      if(_found)
        AddCXXCompilerFlag("/arch:AVX512" FLAGS ARCHITECTURE_CXX_FLAGS RESULT _found)
      endif()
      if(NOT _found)
        _ofa_find(_enable_extension_flag_list "avx2" _found)
        if(_found)
          AddCXXCompilerFlag("/arch:AVX2" FLAGS ARCHITECTURE_CXX_FLAGS RESULT _found)
        endif()
      endif()
      if(NOT _found)
        _ofa_find(_enable_extension_flag_list "avx" _found)
        if(_found)
          AddCXXCompilerFlag("/arch:AVX" FLAGS ARCHITECTURE_CXX_FLAGS RESULT _found)
        endif()
      endif()
      if(NOT _found)
        _ofa_find(_enable_extension_flag_list "sse2" _found)
        if(_found)
          AddCXXCompilerFlag("/arch:SSE2" FLAGS ARCHITECTURE_CXX_FLAGS)
        endif()
      endif()
      if(NOT _found)
        _ofa_find(_enable_extension_flag_list "sse" _found)
        if(_found)
          AddCXXCompilerFlag("/arch:SSE" FLAGS ARCHITECTURE_CXX_FLAGS)
        endif()
      endif()
      foreach(_extension ${_enable_extension_flag_list})
        string(TOUPPER "${_extension}" _extension)
        string(REPLACE "[-.+/:= ]" "_" _extension "__${_extension}__")
        add_definitions("-D${_extension}")
      endforeach(_extension)

    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel"
        OR CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")

      if(WIN32)
        # Intel (on Windows)
        set(OFA_map_knl "-QxKNL;-QxMIC-AVX512")
        set(OFA_map_knm "-QxKNM;-QxMIC-AVX512")
  set(OFA_map_raptorlake "-QxRAPTORLAKE;-QxALDERLAKE;-QxCORE-AVX512")
        set(OFA_map_rocketlake "-QxROCKETLAKE;-QxALDERLAKE;-QxCORE-AVX512")
        set(OFA_map_sapphirerapids "-QxSAPPHIRERAPIDS;-QxALDERLAKE;-QxCORE-AVX512")
        set(OFA_map_alderlake "-QxALDERLAKE;-QxCORE-AVX512")
        set(OFA_map_tigerlake "-QxTIGERLAKE;-QxCORE-AVX512")
        set(OFA_map_icelake-server "-QxICELAKE-SERVER;-QxCORE-AVX512")
        set(OFA_map_icelake-avx512 "-QxICELAKE-SERVER;-QxCORE-AVX512")
        set(OFA_map_icelake-client "-QxICELAKE-CLIENT;-QxCORE-AVX512")
        set(OFA_map_icelake "-QxICELAKE-CLIENT;-QxCORE-AVX512")
        set(OFA_map_cannonlake "-QxCANNONLAKE;-QxCORE-AVX512")
        set(OFA_map_cooperlake "-QxCOOPERLAKE;-QxCORE-AVX512")
        set(OFA_map_cascadelake "-QxCASCADELAKE;-QxCORE-AVX512")
        set(OFA_map_skylake-avx512 "-QxSKYLAKE-AVX512;-QxCORE-AVX512")
        set(OFA_map_skylake "-QxSKYLAKE;-QxCORE-AVX2")
        set(OFA_map_broadwell "-QxBROADWELL;-QxCORE-AVX2")
        set(OFA_map_haswell "-QxHASWELL;-QxCORE-AVX2")
        set(OFA_map_ivybridge "-QxIVYBRIDGE;-QxCORE-AVX-I")
        set(OFA_map_sandybridge "-QxSANDYBRIDGE;-QxAVX")
        set(OFA_map_westmere "-QxSSE4.2")
        set(OFA_map_nehalem "-QxSSE4.2")
        set(OFA_map_penryn "-QxSSSE3")
        set(OFA_map_merom "-QxSSSE3")
        set(OFA_map_core2 "-QxSSE3")
        set(_ok FALSE)
      else()
        # Intel (in Linux)
        set(OFA_map_knl "-xKNL;-xMIC-AVX512")
        set(OFA_map_knm "-xKNM;-xMIC-AVX512")
  set(OFA_map_raptorlake "-xRAPTORLAKE;-xALDERLAKE;-xCORE-AVX512")
        set(OFA_map_rocketlake "-xROCKETLAKE;-xALDERLAKE;-xCORE-AVX512")
        set(OFA_map_sapphirerapids "-xSAPPHIRERAPIDS;-xALDERLAKE;-xCORE-AVX512")
        set(OFA_map_alderlake "-xALDERLAKE;-xCORE-AVX512")
        set(OFA_map_tigerlake "-xTIGERLAKE;-xCORE-AVX512")
        set(OFA_map_icelake-server "-xICELAKE-SERVER;-xCORE-AVX512")
        set(OFA_map_icelake-avx512 "-xICELAKE-SERVER;-xCORE-AVX512")
        set(OFA_map_icelake-client "-xICELAKE-CLIENT;-xCORE-AVX512")
        set(OFA_map_icelake "-xICELAKE-CLIENT;-xCORE-AVX512")
        set(OFA_map_cannonlake "-xCANNONLAKE;-xCORE-AVX512")
        set(OFA_map_cooperlake "-xCOOPERLAKE;-xCORE-AVX512")
        set(OFA_map_cascadelake "-xCASCADELAKE;-xCORE-AVX512")
        set(OFA_map_skylake-avx512 "-xSKYLAKE-AVX512;-xCORE-AVX512")
        set(OFA_map_skylake "-xSKYLAKE;-xCORE-AVX2")
        set(OFA_map_broadwell "-xBROADWELL;-xCORE-AVX2")
        set(OFA_map_haswell "-xHASWELL;-xCORE-AVX2")
        set(OFA_map_ivybridge "-xIVYBRIDGE;-xCORE-AVX-I")
        set(OFA_map_sandybridge "-xSANDYBRIDGE;-xAVX")
        set(OFA_map_westmere "-xSSE4.2")
        set(OFA_map_nehalem "-xSSE4.2")
        set(OFA_map_penryn "-xSSSE3")
        set(OFA_map_merom "-xSSSE3")
        set(OFA_map_core2 "-xSSE3")
        set(_ok FALSE)
      endif()

      foreach(_arch ${_march_flag_list})
        if(DEFINED OFA_map_${_arch})
          foreach(_flag ${OFA_map_${_arch}})
            AddCXXCompilerFlag(${_flag} FLAGS ARCHITECTURE_CXX_FLAGS RESULT _ok)
            if(_ok)
              break()
            endif()
          endforeach()
          if(_ok)
            break()
          endif()
        endif()
      endforeach()
      if(NOT _ok)
        # This is the Intel compiler, so SSE2 is a very reasonable baseline.
        message(STATUS "[OFA] Did not recognize the requested architecture flag ${_arch}, falling back to SSE2")
        if(WIN32)
          AddCXXCompilerFlag("-QxSSE2" FLAGS ARCHITECTURE_CXX_FLAGS)
        else()
          AddCXXCompilerFlag("-xSSE2" FLAGS ARCHITECTURE_CXX_FLAGS)
        endif()
      endif()

      # Set -m<_extension> flag for enabled features
      foreach(_extension ${_enable_extension_flag_list})
        AddCXXCompilerFlag("${_enable_flag}${_extension}" FLAGS ARCHITECTURE_CXX_FLAGS)
      endforeach(_extension)

      # Set -mno-<_extension> flag for disabled features
      if(DEFINED _disable_flag)
        foreach(_extension ${_disable_extension_flag_list})
          AddCXXCompilerFlag("${_disable_flag}${_extension}" FLAGS ARCHITECTURE_CXX_FLAGS)
        endforeach(_extension)
      endif()

    elseif(CMAKE_CXX_COMPILER_ID MATCHES "SunPro")

      # Set -xtarget flag
      foreach(_flag ${_march_flag_list})
        AddCXXCompilerFlag("-xtarget=${_flag}" FLAGS ARCHITECTURE_CXX_FLAGS RESULT _good)
        if(_good)
          break()
        endif(_good)
      endforeach(_flag)

      # Set -xarch=<feature> flag for enabled features
      foreach(_flag ${_enable_extension_flag_list})
        AddCXXCompilerFlag("-xarch=${_flag}" FLAGS ARCHITECTURE_CXX_FLAGS)
      endforeach(_flag)

      # TODO PGI/Cray ...

    else()
      # Others: GNU, Clang and variants

      # Set -march flag
      foreach(_flag ${_march_flag_list})
        AddCXXCompilerFlag("-march=${_flag}" FLAGS ARCHITECTURE_CXX_FLAGS RESULT _good)
        if(_good)
          break()
        endif(_good)
      endforeach(_flag)

      # Set -m<feature> flag for enabled features
      foreach(_flag ${_enable_extension_flag_list})
        AddCXXCompilerFlag("-m${_flag}" FLAGS ARCHITECTURE_CXX_FLAGS)
      endforeach(_flag)

      # Set -mno-feature flag for disabled features
      foreach(_flag ${_disable_extension_flag_list})
        AddCXXCompilerFlag("-mno-${_flag}" FLAGS ARCHITECTURE_CXX_FLAGS)
      endforeach(_flag)
    endif()
  endif()

  # Compile code with profiling instrumentation
  if(TARGET_PROFILER STREQUAL "gprof")
    AddCXXCompilerFlag("-pg" FLAGS ARCHITECTURE_CXX_FLAGS)
  elseif(TARGET_PROFILER STREQUAL "vtune")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      # Need to check if this also works on Windows
      AddCXXCompilerFlag("-g" FLAGS ARCHITECTURE_CXX_FLAGS)
      AddCXXCompilerFlag("-debug inline-debug-info" FLAGS ARCHITECTURE_CXX_FLAGS)
      AddCXXCompilerFlag("-D TBB_USE_THREADING_TOOLS" FLAGS ARCHITECTURE_CXX_FLAGS)
      AddCXXCompilerFlag("-parallel-source-info=2" FLAGS ARCHITECTURE_CXX_FLAGS)
      AddCXXCompilerFlag("-gline-tables-only" FLAGS ARCHITECTURE_CXX_FLAGS)
      AddCXXCompilerFlag("-fdebug-info-for-profiling" FLAGS ARCHITECTURE_CXX_FLAGS)
      AddCXXCompilerFlag("-Xsprofile" FLAGS ARCHITECTURE_CXX_FLAGS)
    endif()
  endif()

  # Remove duplicate flags
  list(REMOVE_DUPLICATES ARCHITECTURE_CXX_FLAGS)

  if(OFA_VERBOSE)
    string(REPLACE ";"  ", " _str "${ARCHITECTURE_CXX_FLAGS}")
    message(STATUS "ARCHITECTURE_CXX_FLAGS: " ${_str})
  endif()

endmacro(OFA_HandleX86Options)
