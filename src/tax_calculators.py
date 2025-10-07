"""
Tax Computation Modules for IncomeTaxGPT
Implements key tax calculations with explanations
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

@dataclass
class TaxCalculation:
    """Result of a tax calculation with explanation"""
    result: float
    breakdown: Dict
    explanation: str
    citations: List[str]
    
    def to_dict(self):
        return {
            'result': self.result,
            'breakdown': self.breakdown,
            'explanation': self.explanation,
            'citations': self.citations
        }

class TaxCalculators:
    """Collection of Indian Income Tax calculators"""
    
    def __init__(self):
        # Tax slabs for FY 2023-24 (you can update these)
        self.old_regime_slabs = [
            (250000, 0),      # Up to 2.5L: 0%
            (500000, 0.05),   # 2.5L to 5L: 5%
            (1000000, 0.20),  # 5L to 10L: 20%
            (float('inf'), 0.30)  # Above 10L: 30%
        ]
        
        self.new_regime_slabs = [
            (300000, 0),      # Up to 3L: 0%
            (600000, 0.05),   # 3L to 6L: 5%
            (900000, 0.10),   # 6L to 9L: 10%
            (1200000, 0.15),  # 9L to 12L: 15%
            (1500000, 0.20),  # 12L to 15L: 20%
            (float('inf'), 0.30)  # Above 15L: 30%
        ]
        
        # Standard deduction
        self.standard_deduction = 50000
        
        # Rebate under 87A
        self.rebate_87a_limit = 500000
        self.rebate_87a_amount = 12500
    
    def calculate_hra_exemption(self, 
                                basic_salary: float,
                                hra_received: float,
                                rent_paid: float,
                                is_metro: bool) -> TaxCalculation:
        """
        Calculate HRA exemption under Section 10(13A)
        
        Args:
            basic_salary: Annual basic salary
            hra_received: Annual HRA received
            rent_paid: Annual rent paid
            is_metro: Whether employee lives in metro city
        
        Returns:
            TaxCalculation with exemption amount and explanation
        """
        # HRA exemption is minimum of three:
        # 1. Actual HRA received
        # 2. 50% (metro) or 40% (non-metro) of basic salary
        # 3. Rent paid - 10% of basic salary
        
        metro_percent = 0.50 if is_metro else 0.40
        
        option1 = hra_received
        option2 = basic_salary * metro_percent
        option3 = max(0, rent_paid - (basic_salary * 0.10))
        
        exemption = min(option1, option2, option3)
        taxable_hra = hra_received - exemption
        
        breakdown = {
            'hra_received': hra_received,
            'option1_actual_hra': option1,
            'option2_percent_of_basic': option2,
            'option3_rent_minus_10_percent': option3,
            'exemption_granted': exemption,
            'taxable_hra': taxable_hra
        }
        
        explanation = f"""
HRA Exemption Calculation (Section 10(13A)):

Inputs:
- Basic Salary: ₹{basic_salary:,.2f}
- HRA Received: ₹{hra_received:,.2f}
- Rent Paid: ₹{rent_paid:,.2f}
- City Type: {'Metro' if is_metro else 'Non-Metro'}

The exemption is the MINIMUM of:
1. Actual HRA received: ₹{option1:,.2f}
2. {int(metro_percent*100)}% of basic salary: ₹{option2:,.2f}
3. Rent paid minus 10% of basic: ₹{option3:,.2f}

Result:
- HRA Exemption: ₹{exemption:,.2f}
- Taxable HRA: ₹{taxable_hra:,.2f}
"""
        
        citations = [
            "Section 10(13A) of Income Tax Act, 1961",
            "Rule 2A of Income Tax Rules, 1962"
        ]
        
        return TaxCalculation(
            result=exemption,
            breakdown=breakdown,
            explanation=explanation.strip(),
            citations=citations
        )
    
    def calculate_80c_deduction(self, investments: Dict[str, float]) -> TaxCalculation:
        """
        Calculate deduction under Section 80C
        
        Args:
            investments: Dictionary of investment types and amounts
                        e.g., {'ppf': 50000, 'elss': 30000, 'life_insurance': 25000}
        
        Returns:
            TaxCalculation with deduction amount
        """
        # Section 80C limit
        limit = 150000
        
        total_invested = sum(investments.values())
        deduction = min(total_invested, limit)
        
        breakdown = {
            'investments': investments,
            'total_invested': total_invested,
            'deduction_limit': limit,
            'deduction_claimed': deduction,
            'excess_amount': max(0, total_invested - limit)
        }
        
        investment_details = '\n'.join([
            f"  - {k.upper()}: ₹{v:,.2f}" 
            for k, v in investments.items()
        ])
        
        explanation = f"""
Section 80C Deduction Calculation:

Eligible Investments:
{investment_details}

Total Invested: ₹{total_invested:,.2f}
Maximum Deduction Limit: ₹{limit:,.2f}

Result:
- Deduction Allowed: ₹{deduction:,.2f}
{"- Excess (not eligible): ₹" + f"{breakdown['excess_amount']:,.2f}" if breakdown['excess_amount'] > 0 else ""}
"""
        
        citations = ["Section 80C of Income Tax Act, 1961"]
        
        return TaxCalculation(
            result=deduction,
            breakdown=breakdown,
            explanation=explanation.strip(),
            citations=citations
        )
    
    def calculate_tax_liability(self,
                               gross_income: float,
                               deductions: Dict[str, float],
                               regime: str = 'old') -> TaxCalculation:
        """
        Calculate tax liability
        
        Args:
            gross_income: Total gross income
            deductions: Dictionary of deductions (only for old regime)
            regime: 'old' or 'new'
        
        Returns:
            TaxCalculation with tax liability
        """
        # Select regime
        slabs = self.old_regime_slabs if regime == 'old' else self.new_regime_slabs
        
        # Calculate taxable income
        if regime == 'old':
            total_deductions = sum(deductions.values()) + self.standard_deduction
        else:
            total_deductions = self.standard_deduction
            deductions = {}  # No deductions in new regime except standard
        
        taxable_income = max(0, gross_income - total_deductions)
        
        # Calculate tax slab-wise
        tax = 0
        prev_limit = 0
        slab_breakdown = []
        
        for limit, rate in slabs:
            if taxable_income <= prev_limit:
                break
            
            taxable_in_slab = min(taxable_income, limit) - prev_limit
            tax_in_slab = taxable_in_slab * rate
            tax += tax_in_slab
            
            if taxable_in_slab > 0:
                slab_breakdown.append({
                    'range': f"₹{prev_limit:,.0f} - ₹{limit:,.0f}" if limit != float('inf') else f"Above ₹{prev_limit:,.0f}",
                    'rate': f"{rate*100:.0f}%",
                    'taxable_amount': taxable_in_slab,
                    'tax': tax_in_slab
                })
            
            prev_limit = limit
        
        # Apply rebate under 87A if eligible
        rebate = 0
        if taxable_income <= self.rebate_87a_limit and regime == 'old':
            rebate = min(tax, self.rebate_87a_amount)
            tax -= rebate
        
        # Add cess (4%)
        cess = tax * 0.04
        total_tax = tax + cess
        
        breakdown = {
            'gross_income': gross_income,
            'deductions': deductions,
            'total_deductions': total_deductions,
            'taxable_income': taxable_income,
            'slab_wise_tax': slab_breakdown,
            'tax_before_rebate': tax + rebate if rebate > 0 else tax,
            'rebate_87a': rebate,
            'tax_after_rebate': tax,
            'cess_4_percent': cess,
            'total_tax_liability': total_tax
        }
        
        slab_details = '\n'.join([
            f"  {s['range']}: ₹{s['taxable_amount']:,.2f} @ {s['rate']} = ₹{s['tax']:,.2f}"
            for s in slab_breakdown
        ])
        
        deduction_details = ""
        if regime == 'old' and deductions:
            deduction_details = "Deductions Applied:\n" + '\n'.join([
                f"  - {k}: ₹{v:,.2f}" for k, v in deductions.items()
            ]) + f"\n  - Standard Deduction: ₹{self.standard_deduction:,.2f}\n\n"
        
        explanation = f"""
Tax Liability Calculation ({regime.upper()} REGIME):

Gross Income: ₹{gross_income:,.2f}
{deduction_details}Total Deductions: ₹{total_deductions:,.2f}
Taxable Income: ₹{taxable_income:,.2f}

Tax Calculation (Slab-wise):
{slab_details}

Subtotal: ₹{tax + rebate if rebate > 0 else tax:,.2f}
{"Rebate u/s 87A: -₹" + f"{rebate:,.2f}" if rebate > 0 else ""}
Tax after rebate: ₹{tax:,.2f}
Health & Education Cess (4%): ₹{cess:,.2f}

TOTAL TAX LIABILITY: ₹{total_tax:,.2f}
"""
        
        citations = [
            "Section 115BAC - New Tax Regime" if regime == 'new' else "Standard Tax Slabs",
            "Section 87A - Rebate for income up to ₹5 lakh" if rebate > 0 else None
        ]
        citations = [c for c in citations if c]
        
        return TaxCalculation(
            result=total_tax,
            breakdown=breakdown,
            explanation=explanation.strip(),
            citations=citations
        )
    
    def compare_regimes(self,
                       gross_income: float,
                       deductions: Dict[str, float]) -> Dict:
        """
        Compare tax liability between old and new regime
        """
        old_calc = self.calculate_tax_liability(gross_income, deductions, 'old')
        new_calc = self.calculate_tax_liability(gross_income, {}, 'new')
        
        savings = old_calc.result - new_calc.result
        better_regime = 'new' if savings > 0 else 'old'
        
        return {
            'old_regime': old_calc.to_dict(),
            'new_regime': new_calc.to_dict(),
            'savings_with_new_regime': savings,
            'better_regime': better_regime,
            'recommendation': f"{'New' if better_regime == 'new' else 'Old'} regime is better by ₹{abs(savings):,.2f}"
        }

# Example usage
if __name__ == "__main__":
    calc = TaxCalculators()
    
    print("="*70)
    print("TAX CALCULATOR EXAMPLES")
    print("="*70)
    
    # Example 1: HRA Calculation
    print("\n1. HRA EXEMPTION CALCULATION")
    print("-" * 70)
    hra_result = calc.calculate_hra_exemption(
        basic_salary=600000,
        hra_received=240000,
        rent_paid=180000,
        is_metro=True
    )
    print(hra_result.explanation)
    
    # Example 2: 80C Deduction
    print("\n\n2. SECTION 80C DEDUCTION")
    print("-" * 70)
    deduction_80c = calc.calculate_80c_deduction({
        'ppf': 50000,
        'elss': 40000,
        'life_insurance': 30000,
        'nsc': 40000
    })
    print(deduction_80c.explanation)
    
    # Example 3: Tax Liability
    print("\n\n3. TAX LIABILITY CALCULATION (OLD REGIME)")
    print("-" * 70)
    tax_old = calc.calculate_tax_liability(
        gross_income=1200000,
        deductions={
            '80C': 150000,
            '80D': 25000
        },
        regime='old'
    )
    print(tax_old.explanation)
    
    # Example 4: Regime Comparison
    print("\n\n4. REGIME COMPARISON")
    print("-" * 70)
    comparison = calc.compare_regimes(
        gross_income=1200000,
        deductions={'80C': 150000, '80D': 25000}
    )
    print(f"Old Regime Tax: ₹{comparison['old_regime']['result']:,.2f}")
    print(f"New Regime Tax: ₹{comparison['new_regime']['result']:,.2f}")
    print(f"\n{comparison['recommendation']}")