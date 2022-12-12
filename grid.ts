import { GImage, GImageSet, Visibility, FIELDS, sort } from "./types.js"
import { StoredVal } from "./storage.js"

const STORE_HIDDEN = new StoredVal('hidden', new Set<String>(), 
                                   (storage) => Array.from(storage), (jsonVal) => new Set(jsonVal))

class GImageGrid {
    imagesetByFilename: Map<string, GImageSet>
    imageByFilename: Map<string, GImage>
    imageSets: Map<string, GImageSet>                 // image sets by key
    imageSetKeys: Array<string>                       // sorted
    fieldUniqueValues: Map<string, Array<Object>>     // unique sorted values for each field
    fieldValueIndex: Map<String, Map<Object, number>> // index in this.fieldUniqueValues for each field, value

    constructor(imageSets: Map<string, GImageSet>) {
        this.update(imageSets)
    }

    update(imageSets: Map<string, GImageSet>) {
        this.imageSets = imageSets
        this.imageSetKeys = sort(imageSets.keys()) as string[]
        this.imagesetByFilename = new Map()
        this.imageByFilename = new Map()

        // set up helper maps
        for (const iset of imageSets.values()) {
            for (const img of iset.images) {
                this.imagesetByFilename.set(img.filename, iset)
                this.imageByFilename.set(img.filename, img)
            }
        }

        // set index values of each image set
        for (const [idx, isetKey] of this.imageSetKeys.entries()) {
            const iset = this.imageSets.get(isetKey)!
            iset.setIdx = idx
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

    isetsForValue(field: string, value: any): Array<GImageSet> {
        const res = new Array<GImageSet>()
        for (const iset of this.imageSets.values()) {
            if (iset[field] == value) {
                res.push(iset)
            }
        }
        return res
    }

    isHidden(field: String, value: any): boolean {
        var key = `${field}/${value}`
        return STORE_HIDDEN.get().has(key)
    }

    setVisibility(field: string, value: any, visibility: Visibility): Visibility {
        var index = this.fieldValueIndex.get(field)?.get(value)
        if (index == undefined) {
            // This happens if we come across a key we don't know about. This happens when
            // imagegrid is run in directory A, but local storage has keys for a different
            // directory "B". IOW, it's not a problem worth logging about.
            // console.log(`can't find index for ${field} ${value}`)
            return "toggle"
        }

        const storageId = `${field}/${value}`
        const curHidden = STORE_HIDDEN.get().has(storageId)
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

        for (const iset of this.isetsForValue(field, value)) {
            iset.visible = !newHidden
        }

        if (newHidden) {
            STORE_HIDDEN.get().add(storageId)
        }
        else {
            STORE_HIDDEN.get().delete(storageId)
        }
        STORE_HIDDEN.save()

        return newHidden ? "hide" : "show"
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


export { GImageGrid, STORE_HIDDEN }
