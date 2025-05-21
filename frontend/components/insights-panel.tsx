"use client"

import { Slider } from "@/components/ui/slider"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface InsightsPanelProps {
  insights: {
    deceptivePercentage: number
    suspiciousPhrases: string[]
    confidenceHeatmap: number[]
  }
  filters: {
    country: string
    modelConfidence: number
    showOnlyDeceptive: boolean
  }
  onFilterChange: (filters: any) => void
}

export default function InsightsPanel({ insights, filters, onFilterChange }: InsightsPanelProps) {
  const updateFilter = (key: string, value: any) => {
    onFilterChange({
      ...filters,
      [key]: value,
    })
  }

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-md shadow-card overflow-hidden">
        <div className="p-4 font-medium text-lg border-b bg-gradient-insight text-white">INSIGHTS</div>

        <div className="p-4">
          <div className="flex justify-between items-center mb-4">
            <div>
              <div className="text-3xl font-bold text-insight-dark">{insights.deceptivePercentage}%</div>
              <div className="text-sm text-neutral">Deceptive Messages</div>
            </div>

            <div>
              <div className="font-medium text-insight-dark">Suspicious Phrases</div>
              <div className="mt-2">
                {insights.suspiciousPhrases.map((phrase, index) => (
                  <div key={index} className="text-sm text-neutral bg-secondary px-2 py-1 rounded mb-1">
                    {phrase}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-6">
            <div className="font-medium mb-2 text-insight-dark">Confidence Heatmap</div>
            <div className="flex items-end h-20 gap-1">
              {insights.confidenceHeatmap.map((value, index) => (
                <div
                  key={index}
                  className="w-full rounded-t transition-all duration-500 ease-in-out"
                  style={{
                    height: `${value * 100}%`,
                    backgroundColor: `var(--chart-${Math.min(Math.ceil(value * 5), 5)})`,
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-md shadow-card overflow-hidden">
        <div className="p-4 font-medium text-lg border-b bg-gradient-insight text-white">FILTERS</div>

        <div className="p-4 space-y-6">
          <div>
            <label className="font-medium block mb-2 text-insight-dark">Country</label>
            <Select value={filters.country} onValueChange={(value) => updateFilter("country", value)}>
              <SelectTrigger className="border-input bg-secondary/50">
                <SelectValue placeholder="Select country" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="All">All</SelectItem>
                <SelectItem value="Italy">Italy</SelectItem>
                <SelectItem value="Germany">Germany</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="font-medium block mb-2 text-insight-dark">Model Confidence</label>
            <div className="px-1">
              <Slider
                value={[filters.modelConfidence]}
                onValueChange={(value) => updateFilter("modelConfidence", value[0])}
                max={50}
                step={1}
                className="py-2"
              />
            </div>
            <div className="flex justify-between text-sm text-neutral mt-1">
              <span>0%</span>
              <span>50%</span>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox
              id="show-deceptive"
              checked={filters.showOnlyDeceptive}
              onCheckedChange={(checked) => updateFilter("showOnlyDeceptive", checked)}
              className="text-primary border-input data-[state=checked]:bg-primary data-[state=checked]:border-primary"
            />
            <label
              htmlFor="show-deceptive"
              className="text-sm font-medium leading-none text-insight-dark peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
            >
              Show only deceptive messages
            </label>
          </div>
        </div>
      </div>
    </div>
  )
}
