## Introduction
This repository serves as a showcasing sample of a huge and comprehensive project, running until 8/31/2024, following Professor Thomas Coleman's methodology to fit and build U.S. Treasury (UST) yield curves from 1925 to the present.

The project focuses on updating the historical fitted forward rates from 1925 onward, transitioning from FORTRAN to Python while enhancing the methodology originally published over 25 years ago in "US Treasury Yield Curves." We utilize monthly UST bond prices sourced from Chicago’s Center for Research in Security Prices (CRSP).

## Repository Structure
- `SystemOutline`: Core structure of the program.
- `YieldProject_Options`: Contains educational materials and methodological details.

## Folders
- `analysis`: Data analysis of UST bonds, including flower bonds, callable bonds, and taxability.
- `src`: Source code for the project.
- `package`: Contains well-developed programs with accompanying unit tests.
- `development`: Programs under development, including main scripts for looping through and displaying rates and returns.
- `test`: Unit testing programs for the source code.
- `output`:
1. Contains the latest 1925-present forward rates, par bond rates, zero bond rates, annuity rates, total return, yield return, and excess yield return tables. These are calculated monthly using the piece-wise constant forward (pwcf) approach with European options and yield-to-worst methods.
2. Animations of par bond rates and forward rates for all months from 1925 to the present.


## References
- Coleman, Thomas S., Lawrence Fisher, and Roger G. Ibbotson. 1993. Historical U.S. Treasury Yield Curves. Moody’s Investors Service.
- Coleman, Thomas S. 1998. “Fitting Forward Rates to Market Data.” SSRN Abstract.
