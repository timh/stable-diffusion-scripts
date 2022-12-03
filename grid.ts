import { GImage, GImageSet, ColumnHeader, Visibility, FIELDS, sort } from "./types.js"
import { StoredVal } from "./storage.js"

const STORE_HIDDEN = new StoredVal('hidden', new Set<String>(), 
                                   (storage) => Array.from(storage), (jsonVal) => new Set(jsonVal))

class GImageGrid {
    imagesetByFilename: Map<string, GImageSet>
    imageSets: Map<string, GImageSet>                 // image sets by key
    imageSetKeys: Array<string>                       // sorted
    fieldUniqueValues: Map<string, Array<Object>>     // unique sorted values for each field
    fieldValueIndex: Map<String, Map<Object, number>> // index in this._fieldUniqueValues for each field, value

    constructor(imageSets: Map<string, GImageSet>) {
        this.update(imageSets)
    }

    update(imageSets: Map<string, GImageSet>) {
        this.imageSets = imageSets
        this.imageSetKeys = sort(imageSets.keys()) as string[]
        this.imagesetByFilename = new Map()
        for (const iset of imageSets.values()) {
            for (const img of iset.images) {
                this.imagesetByFilename.set(img.filename, iset)
            }
        }

        // build sorted list of unique values for each field. start by building a set.
        var uniqueFieldValuesSet = new Map<string, Set<any>>()
        for (const field of FIELDS) {
            var valueSet = new Set<any>()
            uniqueFieldValuesSet.set(field, valueSet)
            for (const iset of imageSets.values()) {
                valueSet.add(iset[field])
            }
        }

        // then convert it to a sorted array
        this.fieldUniqueValues = new Map<string, Array<any>>()
        for (const field of FIELDS) {
            var val = uniqueFieldValuesSet.get(field)!
            this.fieldUniqueValues.set(field, sort(val))
        }

        this.fieldValueIndex = new Map<string, Map<Object, number>>()
        for (const field of FIELDS) {
            var valueMap = new Map<any, number>()
            this.fieldValueIndex.set(field, valueMap)
            for (const [idx, value] of this.fieldUniqueValues.get(field)!.entries()) {
                valueMap.set(value, idx)
            }
        }
    }

    isHidden(field: String, value: any): boolean {
        var key = `${field}/${value}`
        return STORE_HIDDEN.get().has(key)
    }

    setVisibility(field: string, value: any, visibility: Visibility): Visibility {
        var index = this.fieldValueIndex.get(field)?.get(value)
        if (index == undefined) {
            console.log(`can't find index for ${field} ${value}`)
            return "toggle"
        }
    
        var className = `${field}_${index}`
        var spanId = `choice_${className}`
        var span = document.getElementById(spanId)
        if (span != null) {
            var curHidden = this.isHidden(field, value)
            var newHidden: boolean
    
            if (visibility == "hide") {
                newHidden = true
            }
            else if (visibility == "show") {
                newHidden = false
            }
            else {
                newHidden = !curHidden
            }
    
            span.className = newHidden ? "" : "selected"
    
            for (const el of document.getElementsByClassName(className)) {
                if (newHidden) {
                    el.className = el.className + " hidden"
                }
                else {
                    el.className = el.className.replace(" hidden", "")
                }
            }
    
            const storageId = `${field}/${value}`
            if (newHidden) {
                STORE_HIDDEN.get().add(storageId)
            }
            else {
                STORE_HIDDEN.get().delete(storageId)
            }
            STORE_HIDDEN.save()
    
            return newHidden ? "hide" : "show"
        }
        console.log(`can't find span ${spanId}`)
        return "toggle"
    }
    
    loadVisibilityFromStore() {
        var hidden = STORE_HIDDEN.get()
        for (const hiddenStr of hidden) {
            var [field, value] = hiddenStr.split("/") as [string, any]
            if (["modelSteps", "modelSeed", "cfg"].indexOf(field) != -1) {
                value = parseInt(value)
            }
            this.setVisibility(field as string, value, "hide")
        }
    }

    
}


export { GImageGrid }
