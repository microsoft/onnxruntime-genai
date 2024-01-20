include(FetchContent)

if (false) # We don't use GSL anymore
    FetchContent_Declare(GSL
            GIT_REPOSITORY "https://github.com/microsoft/GSL"
            GIT_TAG "v4.0.0"
            GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(GSL)
endif ()

if (false) # We don't use SafeInt anymore
    FetchContent_Declare(safeint
            GIT_REPOSITORY "https://github.com/dcleblanc/SafeInt"
    )

    FetchContent_MakeAvailable(safeint)
endif ()
